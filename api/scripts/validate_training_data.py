#!/usr/bin/env python3
"""Validate and split Decimus LLM training data.

Checks JSONL integrity, reports statistics, and creates train/eval splits.

Usage:
    python scripts/validate_training_data.py [--input data/training/orchestration_instruct.jsonl]
"""

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "training" / "orchestration_instruct.jsonl"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English)."""
    return len(text) // 4


def main():
    parser = argparse.ArgumentParser(description="Validate Decimus LLM training data")
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--split-ratio", type=float, default=0.9,
                        help="Train/eval split ratio (default: 0.9)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found. Run generate_training_data.py first.")
        return 1

    print(f"Validating {args.input}...")
    print()

    records = []
    errors = 0
    empty_fields = 0
    token_counts = []

    with open(args.input, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Line {line_num}: JSON parse error: {e}")
                errors += 1
                continue

            # Check required fields
            instruction = record.get("instruction", "")
            output = record.get("output", "")

            if not instruction or not output:
                empty_fields += 1
                continue

            # Estimate token count for the full training example
            full_text = instruction + record.get("input", "") + output
            tokens = estimate_tokens(full_text)
            token_counts.append(tokens)

            records.append(record)

    print(f"Total lines parsed: {len(records) + errors + empty_fields}")
    print(f"Valid records: {len(records)}")
    print(f"Parse errors: {errors}")
    print(f"Empty instruction/output: {empty_fields}")
    print()

    if not records:
        print("No valid records found!")
        return 1

    # Token statistics
    token_counts.sort()
    avg_tokens = sum(token_counts) / len(token_counts)
    median_tokens = token_counts[len(token_counts) // 2]
    max_tokens = max(token_counts)
    min_tokens = min(token_counts)

    print("Token statistics (estimated):")
    print(f"  Min: {min_tokens}")
    print(f"  Max: {max_tokens}")
    print(f"  Mean: {avg_tokens:.0f}")
    print(f"  Median: {median_tokens}")
    print()

    # Flag outliers
    over_2048 = sum(1 for t in token_counts if t > 2048)
    if over_2048:
        print(f"  WARNING: {over_2048} records exceed 2048 tokens (will be truncated during training)")

    # Check Source E JSON validity
    json_outputs = 0
    json_valid = 0
    expected_keys = {"melody", "bass", "harmony", "countermelody", "advice"}
    for record in records:
        output = record["output"]
        if output.strip().startswith("{"):
            json_outputs += 1
            try:
                parsed = json.loads(output)
                if expected_keys.issubset(set(parsed.keys())):
                    json_valid += 1
            except json.JSONDecodeError:
                pass

    if json_outputs:
        print(f"JSON planning outputs: {json_outputs} total, {json_valid} valid with expected keys")
        print()

    # Split into train/eval
    random.seed(args.seed)
    random.shuffle(records)
    split_idx = int(len(records) * args.split_ratio)
    train_records = records[:split_idx]
    eval_records = records[split_idx:]

    train_path = args.input.with_name("train.jsonl")
    eval_path = args.input.with_name("eval.jsonl")

    with open(train_path, 'w', encoding='utf-8') as f:
        for record in train_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(eval_path, 'w', encoding='utf-8') as f:
        for record in eval_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Train set: {len(train_records)} records -> {train_path}")
    print(f"Eval set:  {len(eval_records)} records -> {eval_path}")
    print()
    print("Validation complete.")
    return 0


if __name__ == "__main__":
    exit(main())
