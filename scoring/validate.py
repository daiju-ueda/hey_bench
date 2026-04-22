"""
症例 YAML を case.schema.json に照らしてバリデーションする lint スクリプト。

使い方:
    python -m scoring.validate cases/
    python -m scoring.validate cases/red_flag/HF-RF-001.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schema" / "case.schema.json"


def load_schema() -> dict:
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def iter_case_files(target: Path):
    if target.is_file():
        yield target
        return
    for path in sorted(target.rglob("*.yaml")):
        yield path
    for path in sorted(target.rglob("*.yml")):
        yield path


def validate_case(path: Path, validator: Draft202012Validator) -> list[str]:
    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return [f"YAML parse error: {e}"]

    errors = []
    for err in sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path)):
        location = "/".join(str(p) for p in err.absolute_path) or "<root>"
        errors.append(f"{location}: {err.message}")

    case_id = (data or {}).get("case_id") if isinstance(data, dict) else None
    expected = path.stem
    if case_id and case_id != expected:
        errors.append(f"case_id '{case_id}' does not match filename stem '{expected}'")

    return errors


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    target = Path(argv[1]).resolve()
    if not target.exists():
        print(f"Target not found: {target}", file=sys.stderr)
        return 2

    schema = load_schema()
    validator = Draft202012Validator(schema)

    total = 0
    failed = 0
    for case_path in iter_case_files(target):
        total += 1
        errors = validate_case(case_path, validator)
        rel = case_path.relative_to(REPO_ROOT)
        if errors:
            failed += 1
            print(f"[FAIL] {rel}")
            for msg in errors:
                print(f"       - {msg}")
        else:
            print(f"[OK]   {rel}")

    print(f"\nTotal: {total}, Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
