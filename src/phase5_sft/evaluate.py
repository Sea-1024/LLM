"""
SFT model evaluation.

Loads an SFT-trained model, generates responses for test prompts,
and computes quality metrics. Supports side-by-side comparison of
base (pretrained) vs SFT model outputs.
"""

import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import torch

from src.phase3_model.model import MiniLLM
from src.phase5_sft.templates import PromptTemplate


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_sft_model(
    model: MiniLLM,
    tokenizer: Any,
    test_data_path: str,
    template_type: str = "alpaca",
    max_new_tokens: int = 256,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """
    Evaluate an SFT model on a test dataset.

    For each test sample:
      1. Format instruction with the prompt template.
      2. Generate a response using model.generate().
      3. Compare generated output with the expected (gold) output.

    Args:
        model: The SFT-trained MiniLLM model.
        tokenizer: Tokenizer with encode/decode methods.
        test_data_path: Path to JSON test data (list of {instruction, input, output}).
        template_type: One of "alpaca", "chatml", "llama".
        max_new_tokens: Maximum tokens to generate per sample.
        device: Torch device. If None, inferred from model.

    Returns:
        A dict with:
            results: list of per-sample dicts
            metrics: aggregated metrics dict
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    template = PromptTemplate(template_type)

    # Load test data
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_samples: list[dict[str, str]] = json.load(f)

    if not test_samples:
        return {"results": [], "metrics": {"error": "Empty test dataset"}}

    results: list[dict[str, Any]] = []
    total_response_len: int = 0
    eos_count: int = 0
    repetition_scores: list[float] = []
    empty_count: int = 0
    generation_times: list[float] = []

    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    for i, sample in enumerate(test_samples):
        instruction = (sample.get("instruction") or "").strip()
        input_text = (sample.get("input") or "").strip()
        expected = (sample.get("output") or "").strip()

        if not instruction:
            results.append({
                "idx": i,
                "instruction": "",
                "expected": expected,
                "generated": "",
                "error": "Empty instruction",
            })
            continue

        # Build the prompt
        prompt, _ = template.format_instruction(instruction, input_text)
        prompt_ids = _encode(tokenizer, prompt)

        # Generate
        t0 = time.time()
        try:
            generated_ids = model.generate(
                torch.tensor([prompt_ids], device=device),
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )
        except Exception as e:
            results.append({
                "idx": i,
                "instruction": instruction,
                "input": input_text,
                "expected": expected,
                "generated": "",
                "error": f"Generation failed: {e}",
            })
            continue

        elapsed = time.time() - t0
        generation_times.append(elapsed)

        # Decode: extract only the new tokens (skip the prompt)
        if isinstance(generated_ids, torch.Tensor):
            generated_ids = generated_ids.tolist()

        # If model.generate returns the full sequence, slice out the prompt
        if len(generated_ids) > len(prompt_ids):
            # Check if the prompt is a prefix
            new_ids = generated_ids[len(prompt_ids):]
        else:
            new_ids = generated_ids

        generated_text = _decode(tokenizer, new_ids)

        # Compute metrics
        resp_len = len(new_ids)
        total_response_len += resp_len

        has_eos = (
            eos_token_id is not None
            and eos_token_id in new_ids
        )
        if has_eos:
            eos_count += 1

        if resp_len == 0:
            empty_count += 1

        rep_score = _repetition_score(generated_text)
        repetition_scores.append(rep_score)

        results.append({
            "idx": i,
            "instruction": instruction,
            "input": input_text,
            "expected": expected,
            "generated": generated_text,
            "response_len": resp_len,
            "has_eos": has_eos,
            "repetition_score": rep_score,
            "generation_time_s": round(elapsed, 3),
        })

    n = len(results)
    metrics: dict[str, Any] = {
        "num_samples": n,
        "avg_response_len": total_response_len / max(n, 1),
        "eos_rate": eos_count / max(n, 1),
        "avg_repetition_score": (
            sum(repetition_scores) / max(len(repetition_scores), 1)
        ),
        "empty_response_rate": empty_count / max(n, 1),
        "avg_generation_time_s": (
            sum(generation_times) / max(len(generation_times), 1)
        ),
    }

    return {"results": results, "metrics": metrics}


# ---------------------------------------------------------------------------
# Bad case analysis
# ---------------------------------------------------------------------------

def analyze_bad_cases(
    results: list[dict[str, Any]],
    threshold: int = 3,
) -> dict[str, Any]:
    """
    Simple heuristic-based bad case classification.

    Classifies each result as:
      - "empty": no response generated
      - "too_short": response shorter than *threshold* tokens
      - "repetitive": repetition score > 0.5
      - "good": otherwise

    Args:
        results: List of per-sample evaluation results.
        threshold: Minimum acceptable response length in tokens.

    Returns:
        Dict with summary counts and per-category samples.
    """
    categories: dict[str, list[dict[str, Any]]] = {
        "empty": [],
        "too_short": [],
        "repetitive": [],
        "good": [],
    }

    for r in results:
        if r.get("error"):
            categories.setdefault("error", []).append(r)
            continue

        resp_len = r.get("response_len", 0)
        rep_score = r.get("repetition_score", 0)

        if resp_len == 0:
            categories["empty"].append(r)
        elif resp_len < threshold:
            categories["too_short"].append(r)
        elif rep_score > 0.5:
            categories["repetitive"].append(r)
        else:
            categories["good"].append(r)

    summary = {
        name: {
            "count": len(items),
            "sample_indices": [it["idx"] for it in items[:10]],
        }
        for name, items in categories.items()
    }

    return {"summary": summary, "details": categories}


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------

@torch.no_grad()
def compare_models(
    base_model: MiniLLM,
    sft_model: MiniLLM,
    tokenizer: Any,
    test_prompts: list[str],
    template_type: str = "alpaca",
    max_new_tokens: int = 256,
    device: Optional[torch.device] = None,
) -> list[dict[str, str]]:
    """
    Side-by-side comparison of base (pretrained) vs SFT model outputs.

    Args:
        base_model: Pretrained (pre-SFT) model.
        sft_model: Post-SFT model.
        tokenizer: Tokenizer with encode/decode.
        test_prompts: List of raw instruction strings.
        template_type: Prompt template type.
        max_new_tokens: Max tokens to generate.
        device: Device.

    Returns:
        List of dicts with keys: prompt, base_output, sft_output.
    """
    if device is None:
        device = next(base_model.parameters()).device

    base_model.eval()
    sft_model.eval()

    template = PromptTemplate(template_type)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    comparisons: list[dict[str, str]] = []

    for prompt_text in test_prompts:
        prompt, _ = template.format_instruction(prompt_text, "")
        prompt_ids = _encode(tokenizer, prompt)
        input_tensor = torch.tensor([prompt_ids], device=device)

        # Base model output
        base_ids = base_model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        if isinstance(base_ids, torch.Tensor):
            base_ids = base_ids.tolist()
        base_new = base_ids[len(prompt_ids):] if len(base_ids) > len(prompt_ids) else base_ids
        base_output = _decode(tokenizer, base_new)

        # SFT model output
        sft_ids = sft_model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        if isinstance(sft_ids, torch.Tensor):
            sft_ids = sft_ids.tolist()
        sft_new = sft_ids[len(prompt_ids):] if len(sft_ids) > len(prompt_ids) else sft_ids
        sft_output = _decode(tokenizer, sft_new)

        comparisons.append({
            "prompt": prompt_text,
            "base_output": base_output,
            "sft_output": sft_output,
        })

    return comparisons


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode(tokenizer: Any, text: str) -> list[int]:
    """Encode text to a list of token IDs."""
    if not text:
        return []
    if hasattr(tokenizer, "encode"):
        result = tokenizer.encode(text)
    else:
        result = tokenizer(text).input_ids
    if hasattr(result, "tolist"):
        return result.tolist()
    return list(result)


def _decode(tokenizer: Any, ids: list[int]) -> str:
    """Decode token IDs to string."""
    if not ids:
        return ""
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(ids)
    return str(ids)


def _repetition_score(text: str, n: int = 4) -> float:
    """
    Compute a simple repetition score based on unique n-grams.

    Low score = diverse text. High score = repetitive.

    Args:
        text: Generated text.
        n: n-gram size.

    Returns:
        Score in [0, 1]; 1.0 means all n-grams are identical.
    """
    if not text or len(text) < n:
        return 0.0

    words = text.split()
    if len(words) < n:
        # Character-level n-grams
        ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]
    else:
        ngrams = [
            " ".join(words[i : i + n])
            for i in range(len(words) - n + 1)
        ]

    if len(ngrams) <= 1:
        return 0.0

    unique = len(set(ngrams))
    total = len(ngrams)
    return 1.0 - (unique / total)


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load SFT model, run evaluation, save report."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate SFT model on test data."
    )
    parser.add_argument(
        "--model_config", type=str, required=True,
        help="Path to model config YAML."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to SFT model checkpoint."
    )
    parser.add_argument(
        "--test_data", type=str, default="data/sft_data/processed/test.json",
        help="Path to test JSON data."
    )
    parser.add_argument(
        "--template_type", type=str, default="alpaca",
        choices=["alpaca", "chatml", "llama"],
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/reports",
        help="Directory for output reports."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
    )
    parser.add_argument(
        "--device", type=str, default="auto",
    )
    parser.add_argument(
        "--compare_base_checkpoint", type=str, default=None,
        help="Optional: path to base (pretrained) checkpoint for comparison."
    )
    args = parser.parse_args()

    # Config
    from src.common.config import MiniLLMConfig
    model_config = MiniLLMConfig.from_yaml(args.model_config)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[evaluate] Device: {device}")

    # Model
    model = MiniLLM(model_config)
    model.to(device)

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[evaluate] Loaded checkpoint from {args.checkpoint}")

    # Tokenizer
    ckpt_dir = os.path.dirname(args.checkpoint)
    tokenizer_path = os.path.join(ckpt_dir, "tokenizer.pkl")
    tokenizer = None
    if os.path.exists(tokenizer_path):
        import pickle
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        print(f"[evaluate] Loaded tokenizer from {tokenizer_path}")
    else:
        # Fallback: dummy tokenizer for testing
        class _DummyTok:
            eos_token_id = 2
            def encode(self, t): return [ord(c) % 30000 for c in t]
            def decode(self, ids): return "".join(chr(i % 256) for i in ids)
        tokenizer = _DummyTok()
        print("[evaluate] WARNING: using dummy tokenizer")

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate
    print(f"[evaluate] Evaluating on {args.test_data} ...")
    eval_result = evaluate_sft_model(
        model=model,
        tokenizer=tokenizer,
        test_data_path=args.test_data,
        template_type=args.template_type,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )

    # Bad case analysis
    bad_cases = analyze_bad_cases(eval_result["results"])

    # Print summary
    metrics = eval_result["metrics"]
    print("\n" + "=" * 50)
    print(" SFT Evaluation Summary")
    print("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n Bad Case Analysis:")
    for cat, info in bad_cases["summary"].items():
        print(f"  {cat}: {info['count']}")

    # Save report
    report = {
        "metrics": metrics,
        "bad_cases_summary": bad_cases["summary"],
        "results": eval_result["results"],
    }
    report_path = os.path.join(args.output_dir, "sft_eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[evaluate] Report saved to {report_path}")

    # Optional: compare with base model
    if args.compare_base_checkpoint:
        print(f"\n[evaluate] Comparing with base model: {args.compare_base_checkpoint}")
        base_model = MiniLLM(model_config)
        base_model.to(device)
        base_ckpt = torch.load(args.compare_base_checkpoint, map_location=device)
        base_state = base_ckpt.get("model_state_dict", base_ckpt.get("state_dict", base_ckpt))
        base_model.load_state_dict(base_state, strict=False)
        base_model.eval()

        # Use first 10 instructions as comparison prompts
        with open(args.test_data, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        prompts = [s.get("instruction", "") for s in test_data[:10]]

        comparisons = compare_models(
            base_model=base_model,
            sft_model=model,
            tokenizer=tokenizer,
            test_prompts=prompts,
            template_type=args.template_type,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        comp_path = os.path.join(args.output_dir, "model_comparison.json")
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comparisons, f, ensure_ascii=False, indent=2)
        print(f"[evaluate] Comparison saved to {comp_path}")

    print("[evaluate] Done.")


if __name__ == "__main__":
    main()
