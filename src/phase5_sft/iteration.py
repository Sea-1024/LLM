"""
Iteration helper utilities for SFT.

Supports data quality analysis, issue classification, and dataset merging
to help guide iterative improvements to SFT training.
"""

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Issue classification
# ---------------------------------------------------------------------------

def classify_issue(generated: str, expected: str) -> str:
    """
    Classify the type of issue in a generated response compared to the expected output.

    Classification categories:
      - "hallucination": Generated text contains factual assertions not in expected.
      - "misunderstanding": Generated text addresses a different topic/question.
      - "truncated": Generated text ends abruptly (no EOS / incomplete sentence).
      - "repetitive": High degree of repetition in generated text.
      - "format_error": Generated text has wrong formatting.
      - "empty": No response generated.
      - "other": None of the above, or unable to determine.

    Args:
        generated: The model-generated response text.
        expected: The gold/reference response text.

    Returns:
        One of the classification strings above.
    """
    gen = (generated or "").strip()
    exp = (expected or "").strip()

    # Empty
    if not gen:
        return "empty"

    # Repetitive
    if _is_repetitive(gen):
        return "repetitive"

    # Truncated (ends with partial sentence, no period/question mark/exclamation)
    if _is_truncated(gen):
        return "truncated"

    # Format error: check for common formatting issues
    if _has_format_error(gen):
        return "format_error"

    # Hallucination vs Misunderstanding (heuristic, word overlap based)
    gen_words = set(_tokenize(gen.lower()))
    exp_words = set(_tokenize(exp.lower()))

    if exp_words:
        overlap = len(gen_words & exp_words) / len(exp_words)
    else:
        overlap = 0.0

    if overlap < 0.15:
        # Very low overlap suggests misunderstanding (different topic)
        return "misunderstanding"

    # Check for hallucination markers: specific numbers, names, facts not in expected
    if _has_hallucination_markers(gen, exp):
        return "hallucination"

    return "other"


def _is_repetitive(text: str, threshold: float = 0.5) -> bool:
    """Check if text is highly repetitive."""
    words = text.split()
    if len(words) < 4:
        return False
    n = min(3, len(words) // 2)
    ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    if len(ngrams) <= 1:
        return False
    unique_ratio = len(set(ngrams)) / len(ngrams)
    return unique_ratio < threshold


def _is_truncated(text: str) -> bool:
    """Check if text appears to be truncated (incomplete sentence)."""
    text = text.strip()
    if not text:
        return False
    # Ends with a mid-sentence fragment
    if not text.endswith((".", "!", "?", ")", '"', "'", ":", "]", "}")):
        # Check if last "word" is very short (likely truncated)
        last_word = text.split()[-1] if text.split() else ""
        if len(last_word) <= 2 and not last_word.isalpha():
            return True
    return False


def _has_format_error(text: str) -> bool:
    """
    Check for common formatting issues:
    - Excessive newlines
    - Mixed markdown/html artifacts
    - Repeated punctuation
    """
    if text.count("\n\n\n") > 2:
        return True
    if re.search(r"[.!?]{4,}", text):
        return True
    if re.search(r"<[^>]+>", text) and re.search(r"</[^>]+>", text):
        # Contains HTML tags (might be correct for some tasks, flag as warning)
        return True
    return False


def _has_hallucination_markers(gen: str, exp: str) -> bool:
    """
    Heuristic hallucination detection: numbers, proper nouns, or specific
    entities in the generated text that are not present in the expected text.
    """
    # Extract numbers
    gen_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", gen))
    exp_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", exp))
    extra_numbers = gen_numbers - exp_numbers
    if len(extra_numbers) > 2:
        return True

    # Extract capitalized words (potential named entities)
    gen_caps = set(re.findall(r"\b[A-Z][a-z]{2,}\b", gen))
    exp_caps = set(re.findall(r"\b[A-Z][a-z]{2,}\b", exp))
    extra_caps = gen_caps - exp_caps
    # Only flag if there are several unexpected capitalized words
    if len(extra_caps) > 3 and len(extra_caps) > len(gen_caps) * 0.5:
        return True

    return False


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------------------------
# Data diversity analysis
# ---------------------------------------------------------------------------

def compute_data_diversity(samples: list[dict[str, str]]) -> dict[str, Any]:
    """
    Compute diversity metrics for a set of SFT samples.

    Metrics computed:
      - num_samples: Total number of samples.
      - avg_instruction_len: Average character length of instructions.
      - avg_output_len: Average character length of outputs.
      - instruction_len_distribution: Histogram of instruction lengths.
      - task_type_distribution: Rough task type classification based on
        instruction keywords.

    Args:
        samples: List of {"instruction": ..., "input": ..., "output": ...} dicts.

    Returns:
        Diversity metrics dict.
    """
    if not samples:
        return {"num_samples": 0, "error": "No samples provided"}

    total_inst_len = 0
    total_out_len = 0
    inst_lens: list[int] = []
    out_lens: list[int] = []
    task_types: Counter = Counter()

    # Task type keywords
    task_keywords = {
        "generation": [
            "write", "create", "generate", "compose", "draft", "make",
        ],
        "classification": [
            "classify", "categorize", "label", "is this", "what type",
        ],
        "qa": [
            "what", "who", "when", "where", "why", "how", "explain",
            "describe", "define", "answer",
        ],
        "summarization": [
            "summarize", "summary", "tl;dr", "briefly", "in short",
        ],
        "translation": [
            "translate", "in french", "in spanish", "in german",
        ],
        "math": [
            "calculate", "solve", "compute", "equation", "math",
        ],
        "code": [
            "code", "program", "function", "python", "javascript",
            "implement", "script", "algorithm",
        ],
        "editing": [
            "rewrite", "edit", "fix", "correct", "improve", "revise",
            "proofread",
        ],
    }

    for sample in samples:
        instruction = (sample.get("instruction") or "").strip()
        output = (sample.get("output") or "").strip()

        inst_len = len(instruction)
        out_len = len(output)
        total_inst_len += inst_len
        total_out_len += out_len
        inst_lens.append(inst_len)
        out_lens.append(out_len)

        # Classify task type
        instr_lower = instruction.lower()
        matched = False
        for task_type, keywords in task_keywords.items():
            if any(kw in instr_lower for kw in keywords):
                task_types[task_type] += 1
                matched = True
                break
        if not matched:
            task_types["other"] += 1

    n = len(samples)

    # Build instruction length histogram (10 bins)
    if inst_lens:
        min_len = min(inst_lens)
        max_len = max(inst_lens)
    else:
        min_len = max_len = 0

    if max_len > min_len:
        bin_width = (max_len - min_len) / 10.0
        hist = [0] * 10
        for length in inst_lens:
            bin_idx = min(int((length - min_len) / bin_width), 9)
            hist[bin_idx] += 1

        hist_bins = [
            {
                "range": (
                    round(min_len + i * bin_width),
                    round(min_len + (i + 1) * bin_width),
                ),
                "count": hist[i],
            }
            for i in range(10)
        ]
    else:
        hist_bins = [{"range": (min_len, max_len), "count": n}]

    return {
        "num_samples": n,
        "avg_instruction_len": total_inst_len / n,
        "avg_output_len": total_out_len / n,
        "instruction_len_distribution": hist_bins,
        "task_type_distribution": dict(task_types.most_common()),
    }


# ---------------------------------------------------------------------------
# Dataset merging
# ---------------------------------------------------------------------------

def merge_datasets(
    dataset_paths: list[str],
    output_path: str,
    deduplicate: bool = True,
) -> int:
    """
    Merge multiple SFT dataset JSON files into one.

    Args:
        dataset_paths: List of paths to JSON files containing SFT samples.
        output_path: Path to write the merged dataset.
        deduplicate: If True, remove duplicate (instruction, input) pairs.

    Returns:
        Total number of samples in the merged dataset.
    """
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for path in dataset_paths:
        if not os.path.exists(path):
            print(f"[merge_datasets] WARNING: file not found, skipping: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(
                f"[merge_datasets] WARNING: {path} is not a list, skipping"
            )
            continue

        added = 0
        for sample in data:
            instruction = (sample.get("instruction") or "").strip()
            input_text = (sample.get("input") or "").strip()
            output_text = (sample.get("output") or "").strip()

            if not instruction or not output_text:
                continue

            if deduplicate:
                key = (instruction, input_text)
                if key in seen:
                    continue
                seen.add(key)

            merged.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
            })
            added += 1

        print(f"[merge_datasets] Loaded {path}: {added} samples added")

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(
        f"[merge_datasets] Merged {len(merged)} samples "
        f"from {len(dataset_paths)} files -> {output_path}"
    )
    return len(merged)


# ---------------------------------------------------------------------------
# Main entry-point (CLI for testing)
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for iteration utilities."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SFT iteration utilities."
    )
    sub = parser.add_subparsers(dest="command")

    # diversity
    div = sub.add_parser("diversity", help="Compute data diversity metrics.")
    div.add_argument("--data_path", type=str, required=True)

    # classify
    clf = sub.add_parser("classify", help="Classify a single generation issue.")
    clf.add_argument("--generated", type=str, required=True)
    clf.add_argument("--expected", type=str, required=True)

    # merge
    mrg = sub.add_parser("merge", help="Merge multiple datasets.")
    mrg.add_argument("--inputs", type=str, nargs="+", required=True)
    mrg.add_argument("--output", type=str, required=True)
    mrg.add_argument("--no_dedup", action="store_true")

    args = parser.parse_args()

    if args.command == "diversity":
        with open(args.data_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        result = compute_data_diversity(samples)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "classify":
        issue = classify_issue(args.generated, args.expected)
        print(f"Issue: {issue}")

    elif args.command == "merge":
        n = merge_datasets(
            args.inputs, args.output, deduplicate=not args.no_dedup
        )
        print(f"Merged {n} samples.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
