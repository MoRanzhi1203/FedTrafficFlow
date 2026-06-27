"""Read-only keyword audit for review comments and meeting alignment."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
from pathlib import Path


TEXT_SUFFIXES = {".md", ".txt", ".rst", ".tex", ".csv", ".json", ".py", ".ipynb"}
EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "results",
    "data",
}
EXCLUDED_PATTERNS = {"*.pt", "*.parquet", "*.png", "*.jpg", "*.jpeg", "*.pdf", "*.webp", "*.gif"}

KEYWORD_GROUPS = {
    "review": ["一审", "审稿", "修改意见", "外审", "review", "revision", "response"],
    "real_data": ["真实数据", "real data", "real-data", "实际数据", "实证", "主实验", "消融", "对比实验", "baseline", "NaiveLastValue"],
    "federated_client": ["联邦学习", "FedAvg", "客户端", "client", "non-IID", "非IID", "异质性", "heterogeneity", "聚合"],
    "model": ["CCN", "CNN", "CNN-LSTM", "LSTM", "Attention", "CNN-LSTM-Attention"],
    "object": ["网格", "路口", "单路口", "区域", "cluster", "region", "grid cell", "intersection"],
    "meeting": ["老师", "导师", "会议", "会议记录", "纪要", "meeting"],
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only alignment audit for review and meeting notes.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--output-json", type=str, default="")
    return parser


def should_skip(path: Path) -> bool:
    if any(part in EXCLUDED_DIRS for part in path.parts):
        return True
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return True
    return any(fnmatch.fnmatch(path.name.lower(), pattern.lower()) for pattern in EXCLUDED_PATTERNS)


def compile_patterns() -> dict[str, re.Pattern[str]]:
    patterns: dict[str, re.Pattern[str]] = {}
    for group_name, keywords in KEYWORD_GROUPS.items():
        escaped = [re.escape(keyword) for keyword in keywords]
        patterns[group_name] = re.compile("|".join(escaped), re.IGNORECASE)
    return patterns


def load_text(path: Path) -> list[str]:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=encoding).splitlines()
        except UnicodeDecodeError:
            continue
    return []


def find_matches(root: Path) -> dict[str, object]:
    patterns = compile_patterns()
    findings: list[dict[str, object]] = []
    file_summaries: list[dict[str, object]] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or should_skip(path):
            continue
        lines = load_text(path)
        if not lines:
            continue

        file_hits: list[dict[str, object]] = []
        for line_no, line in enumerate(lines, start=1):
            matched_groups = [group for group, pattern in patterns.items() if pattern.search(line)]
            if not matched_groups:
                continue
            snippet = line.strip()
            if len(snippet) > 220:
                snippet = snippet[:217] + "..."
            file_hits.append(
                {
                    "line_no": line_no,
                    "groups": matched_groups,
                    "snippet": snippet,
                }
            )

        if not file_hits:
            continue

        relative_path = path.relative_to(root).as_posix()
        file_summaries.append(
            {
                "path": relative_path,
                "hit_count": len(file_hits),
                "groups": sorted({group for hit in file_hits for group in hit["groups"]}),
            }
        )
        findings.extend(
            {
                "path": relative_path,
                "line_no": hit["line_no"],
                "groups": hit["groups"],
                "snippet": hit["snippet"],
            }
            for hit in file_hits
        )

    return {"files": file_summaries, "findings": findings}


def main() -> None:
    args = build_arg_parser().parse_args()
    root = Path(args.root).resolve()
    audit = find_matches(root)
    print(f"[audit_root] {root}")
    print(f"[matched_files] {len(audit['files'])}")
    print(f"[matched_lines] {len(audit['findings'])}")
    for item in audit["files"][:40]:
        print(f"- {item['path']} :: {','.join(item['groups'])} :: hits={item['hit_count']}")
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[output_json] {output_path}")


if __name__ == "__main__":
    main()
