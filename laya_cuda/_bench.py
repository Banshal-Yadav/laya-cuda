"""CLI benchmark entry point."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Benchmark laya-cuda inference speed")
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--n", type=int, default=50, help="Number of timed runs")
    args = parser.parse_args()

    from laya_cuda import load

    print(f"Loading {args.model} ...")
    agent = load(args.model, device=args.device, dtype=args.dtype, compile=args.compile)

    state = "I was billed twice for the same subscription. Please refund the duplicate charge."
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Who should handle this?",
            "criteria": ["billing", "technical", "sales"],
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "Is this time-sensitive?",
        },
    }

    print(f"\nWarm-up inference...")
    result = agent.predict(state, questions)
    print(f"Answers: {result['answers']}")

    print(f"\nBenchmarking {args.n} runs...")
    stats = agent.benchmark(state, questions, n=args.n)
    print(f"\n{'='*40}")
    print(f"Device:  {stats['device']}")
    print(f"Dtype:   {stats['dtype']}")
    print(f"P50:     {stats['p50_ms']:.2f} ms")
    print(f"P95:     {stats['p95_ms']:.2f} ms")
    print(f"Mean:    {stats['mean_ms']:.2f} ms")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
