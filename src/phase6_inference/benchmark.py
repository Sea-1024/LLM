"""
Benchmark inference speed and memory usage.

Measures tokens-per-second, latency, and memory footprint
across multiple test prompts.
"""
import time
import torch
import psutil
import os
from typing import List, Dict, Any


def benchmark_inference(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: List[str],
    num_runs: int = 3,
    max_tokens: int = 128,
) -> Dict[str, Any]:
    """
    Benchmark generation speed across multiple prompts.

    Each prompt is run `num_runs` times (with a warmup run first).
    Results are aggregated across prompts.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompts: List of test prompts.
        num_runs: Number of timed runs per prompt (after warmup).
        max_tokens: Maximum tokens to generate per run.

    Returns:
        A dict with:
            - avg_tokens_per_second: float
            - min_tokens_per_second: float
            - max_tokens_per_second: float
            - avg_latency_ms: float
            - memory_mb: float
            - detailed_results: list of per-prompt stats
    """
    from src.phase6_inference.generate import TextGenerator

    generator = TextGenerator(model, tokenizer)

    results: List[Dict[str, Any]] = []

    for prompt in prompts:
        prompt_times: List[float] = []
        prompt_tokens: List[int] = []

        for _run_idx in range(num_runs + 1):  # +1 for warmup
            # Warmup is not timed
            is_warmup = (_run_idx == 0)
            result = generator.generate(prompt, max_new_tokens=max_tokens)

            if is_warmup:
                continue

            elapsed = 0.0  # will be measured below
            prompt_tokens.append(result["num_tokens"])

        # Re-run with proper timing
        prompt_times = []
        prompt_tokens = []
        for _run_idx in range(num_runs):
            # Warmup before each timed run for consistency
            _ = generator.generate(prompt, max_new_tokens=max(1, max_tokens // 4))

            start = time.time()
            result = generator.generate(prompt, max_new_tokens=max_tokens)
            elapsed = time.time() - start

            prompt_times.append(elapsed)
            prompt_tokens.append(result["num_tokens"])

        avg_time = sum(prompt_times) / len(prompt_times)
        avg_tokens_val = sum(prompt_tokens) / len(prompt_tokens)

        results.append({
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "avg_time_s": avg_time,
            "avg_tokens": avg_tokens_val,
            "tokens_per_second": avg_tokens_val / avg_time if avg_time > 0 else 0.0,
        })

    # ---- Aggregate stats ----
    all_tps = [r["tokens_per_second"] for r in results]
    all_times = [r["avg_time_s"] for r in results]

    # ---- Memory usage ----
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)

    return {
        "avg_tokens_per_second": sum(all_tps) / len(all_tps) if all_tps else 0.0,
        "min_tokens_per_second": min(all_tps) if all_tps else 0.0,
        "max_tokens_per_second": max(all_tps) if all_tps else 0.0,
        "avg_latency_ms": (sum(all_times) / len(all_times)) * 1000.0 if all_times else 0.0,
        "memory_mb": memory_mb,
        "detailed_results": results,
    }


def print_benchmark_report(benchmark_result: Dict[str, Any]) -> None:
    """
    Pretty-print benchmark results to the console.

    Args:
        benchmark_result: The dict returned by `benchmark_inference`.
    """
    print("\n" + "=" * 50)
    print("INFERENCE BENCHMARK REPORT")
    print("=" * 50)
    print(f"Avg tokens/second:  {benchmark_result['avg_tokens_per_second']:.1f}")
    print(f"Min tokens/second:  {benchmark_result['min_tokens_per_second']:.1f}")
    print(f"Max tokens/second:  {benchmark_result['max_tokens_per_second']:.1f}")
    print(f"Avg latency:        {benchmark_result['avg_latency_ms']:.0f} ms")
    print(f"Memory usage:       {benchmark_result['memory_mb']:.0f} MB")
    print("-" * 50)
    for r in benchmark_result["detailed_results"]:
        print(f"  [{r['tokens_per_second']:.1f} tok/s] {r['prompt']}")
    print("=" * 50)


def main() -> None:
    """
    Entry point: loads model, runs benchmark, prints report.

    Usage:
        python -m src.phase6_inference.benchmark \
            --model_config configs/model/config_25m.yaml \
            --model_path output/final/model.safetensors \
            --tokenizer_path models/tokenizer/tokenizer.json
    """
    import argparse
    parser = argparse.ArgumentParser(description="MiniLLM Inference Benchmark")
    parser.add_argument(
        "--model_config",
        default="configs/model/config_25m.yaml",
        help="Path to model YAML config",
    )
    parser.add_argument(
        "--model_path",
        default="output/final/model.safetensors",
        help="Path to safetensors model weights",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="models/tokenizer/tokenizer.json",
        help="Path to tokenizer JSON file",
    )
    args = parser.parse_args()

    from src.common.config import MiniLLMConfig
    from src.phase3_model.model import MiniLLM
    from src.common.utils import load_tokenizer
    from safetensors.torch import load_file

    print("Loading model configuration...")
    config = MiniLLMConfig.from_yaml(args.model_config)

    print("Loading model weights...")
    model = MiniLLM(config)
    state_dict = load_file(args.model_path)
    model.load_state_dict(state_dict)
    model.eval()

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer_path)

    test_prompts = [
        "Explain what machine learning is in simple terms.",
        "Write a short poem about spring.",
        "What is the capital of France?",
        "def fibonacci(n):",
        "The three primary colors are",
        "Translate 'hello' to Chinese:",
    ]

    result = benchmark_inference(model, tokenizer, test_prompts)
    print_benchmark_report(result)


if __name__ == "__main__":
    main()
