from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.research_attention_audit import (  # noqa: E402
    audit_attention_rows,
    read_attention_files,
)


# Preserve the original public tool function used by tests and prior scripts.
audit_rows = audit_attention_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-files", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collapse-threshold", type=float, default=0.95)
    parser.add_argument("--max-collapse-rate", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, inputs = read_attention_files(args.fold_files)
    report = {
        "inputs": inputs,
        **audit_attention_rows(
            rows,
            collapse_threshold=args.collapse_threshold,
            max_collapse_rate=args.max_collapse_rate,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
