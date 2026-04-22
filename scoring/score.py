"""
静的症例採点パイプライン(v0.1 雛形)。

使い方:
    python -m scoring.score \
        --cases cases/ \
        --responses runs/2026-04-22_ai-v0/ \
        --out runs/2026-04-22_ai-v0/scores.csv

responses ディレクトリには各症例に対応する AI 応答 JSON/YAML を
`<case_id>.json` または `<case_id>.yaml` で配置する。

AI 応答スキーマは scoring/rubric.py のモジュール docstring 参照。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import yaml

from scoring.rubric import CaseScore, score_case, JudgeFn


def load_case(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_response(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".json",):
        return json.loads(text)
    return yaml.safe_load(text)


def find_response(responses_dir: Path, case_id: str) -> Optional[Path]:
    for ext in (".json", ".yaml", ".yml"):
        cand = responses_dir / f"{case_id}{ext}"
        if cand.exists():
            return cand
    return None


def load_judge_fn(spec: Optional[str]) -> Optional[JudgeFn]:
    """
    judge_fn を動的にロードする。spec は 'package.module:callable' 形式。
    v0.1 では未接続(None)運用を基本とし、v0.2 以降で LLM judge 実装を差し込む。
    """
    if not spec:
        return None
    import importlib
    module_path, _, attr = spec.partition(":")
    if not module_path or not attr:
        raise SystemExit(f"invalid --judge spec: {spec!r} (expected 'module:callable')")
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


def aggregate(rows: list[CaseScore]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    safety_pass = sum(1 for r in rows if r.safety_pass)
    case_pass = sum(1 for r in rows if r.case_pass)
    must_ask_mean = sum(r.must_ask_coverage for r in rows) / n
    return {
        "n": n,
        "safety_pass_rate": safety_pass / n,
        "safety_gate_violations": n - safety_pass,
        "case_pass_rate": case_pass / n,
        "must_ask_coverage_mean": round(must_ask_mean, 3),
    }


def write_csv(rows: list[CaseScore], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "safety_pass",
        "safety_reasons",
        "must_ask_coverage",
        "must_ask_miss",
        "plan_score",
        "explanation_score",
        "dialogue_score",
        "case_pass",
        "notes",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_row())


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path,
                        help="症例 YAML のディレクトリまたは単一ファイル")
    parser.add_argument("--responses", required=True, type=Path,
                        help="AI 応答の JSON/YAML を配置したディレクトリ")
    parser.add_argument("--out", required=True, type=Path,
                        help="出力 CSV パス")
    parser.add_argument("--judge", default=None,
                        help="LLM judge 関数の import spec (例: 'my_pkg.judge:run')。未指定時は未接続。")
    args = parser.parse_args(argv[1:])

    judge_fn = load_judge_fn(args.judge)

    case_paths: list[Path]
    if args.cases.is_file():
        case_paths = [args.cases]
    else:
        case_paths = sorted(args.cases.rglob("*.yaml")) + sorted(args.cases.rglob("*.yml"))

    if not case_paths:
        print(f"No cases found under {args.cases}", file=sys.stderr)
        return 2

    results: list[CaseScore] = []
    for cp in case_paths:
        case = load_case(cp)
        case_id = case.get("case_id")
        rp = find_response(args.responses, case_id) if case_id else None
        if rp is None:
            results.append(
                CaseScore(
                    case_id=case_id or cp.stem,
                    safety_pass=False,
                    safety_reasons=["AI 応答ファイルが見つからない"],
                    notes=["response missing"],
                )
            )
            continue
        response = load_response(rp)
        results.append(score_case(case, response, judge_fn=judge_fn))

    write_csv(results, args.out)

    agg = aggregate(results)
    print("=== Aggregate ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")

    failed = sum(1 for r in results if not r.case_pass)
    print(f"\nCases: {len(results)}, Pass: {len(results) - failed}, Fail: {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
