import argparse
import json
from pathlib import Path

from eval import eval_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--evaluation_file_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()

    evaluation_file_path = str(Path(args.evaluation_file_path).resolve())
    scores, detailed_scores = eval_score(args.set_type, file_path=evaluation_file_path)

    payload = {
        "set_type": args.set_type,
        "evaluation_file_path": evaluation_file_path,
        "scores": scores,
        "detailed_scores": detailed_scores,
    }

    for key, value in scores.items():
        print(f"{key}: {value * 100}%")

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
