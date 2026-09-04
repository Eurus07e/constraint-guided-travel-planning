"""Evaluate an indexed subset with the same protocol as the full evaluator."""
import argparse
import json
from pathlib import Path
from evaluation.protocol import criterion_pass as _criterion_pass, summarize_cases, evaluate_file


def evaluate(input_path, set_type):
    return evaluate_file(input_path, set_type)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--set-type', choices=('train','validation','test'), default='validation')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.input, args.set_type)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
