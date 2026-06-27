"""Lightweight markdown helpers for region-full-cells reports."""

from __future__ import annotations


def pipe_table(headers: list[str], rows: list[list[object]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "|" + "|".join(["---"] * len(headers)) + "|"
    body_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, divider_line, *body_lines])


def fmt(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return "未运行 / 目录不存在"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"

