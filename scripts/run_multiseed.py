#!/usr/bin/env python3
"""
Helper script to run temperature forecasting experiments with multiple seeds.
This is required for Deliverable 2, which asks to train and test each model
with 5 different seeds.

Usage:
    python scripts/run_multiseed.py --model tcn --seeds 5
    python scripts/run_multiseed.py --model enhanced_rnn --dataset temperature --seeds 5
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_experiment(model, dataset, seed, additional_args=None):
    """Run a single experiment with specified model, dataset, and seed."""
    cmd = [
        sys.executable, "-m", "experiments.run_temperature_prediction",
        f"dataset={dataset}",
        f"model={model}",
        f"seed={seed}",
        f"experiment_name=Temperature_{model}_seed{seed}"
    ]

    if additional_args:
        cmd.extend(additional_args)

    print(f"\n{'='*80}")
    print(f"Running: {model} with seed {seed}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode != 0:
        print(f"WARNING: Experiment failed for {model} with seed {seed}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run temperature forecasting experiments with multiple seeds"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["tcn", "enhanced_rnn", "improved_stgnn", "attn_longterm", "pers_st", "icon"],
        help="Model to train"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="temperature",
        help="Dataset configuration to use"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds to run (default: 5)"
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="Starting seed value (default: 0)"
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Additional arguments to pass to the experiment script"
    )

    args = parser.parse_args()

    # Run experiments with different seeds
    successes = 0
    failures = 0

    for i in range(args.seeds):
        seed = args.start_seed + i
        success = run_experiment(
            args.model,
            args.dataset,
            seed,
            args.extra_args
        )
        if success:
            successes += 1
        else:
            failures += 1

    # Summary
    print(f"\n{'='*80}")
    print(f"Multi-seed experiment summary for {args.model}:")
    print(f"  Successful: {successes}/{args.seeds}")
    print(f"  Failed: {failures}/{args.seeds}")
    print(f"{'='*80}\n")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
