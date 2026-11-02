#!/usr/bin/env python3
"""Render a reviewer CSV as a readable, identity-blind offline HTML packet."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def _block(title: str, value: str, css_class: str = "") -> str:
    return (
        f'<section class="block {css_class}"><h3>{html.escape(title)}</h3>'
        f'<div class="text">{html.escape(value).replace(chr(10), "<br>")}</div></section>'
    )


def render(source: Path, output: Path) -> None:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no review rows in {source}")
    cards: list[str] = []
    for row in rows:
        answers = "".join(_block(f"候选 {code}", row[f"answer_{code}"]) for code in "ABCD")
        cards.append(
            '<article class="case">'
            f'<h2>病例 {html.escape(row["display_order"])} · '
            f'{html.escape(row["blind_item_id"])}</h2>'
            + _block("统一病例与作答任务", row["task_text_private"], "task")
            + answers
            + '<section class="rating"><strong>请在评分表同一 blind_item_id 行填写：</strong>'
              ' best_choice；如并列填写 tied_candidates；unsafe_candidates 必填（无则 NONE）；'
              '再填写 primary_reason 与 confidence_1_3。</section>'
            + "</article>"
        )
    document = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>中医病例智能体专家盲审阅读稿</title>
<style>
body{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;background:#f3f4f6;color:#111827;margin:0}
main{max-width:1040px;margin:0 auto;padding:28px}
.intro,.case{background:#fff;border:1px solid #d1d5db;border-radius:12px;padding:22px;margin:0 0 24px;box-shadow:0 2px 8px #0000000d}
h1{margin:0 0 10px;font-size:26px} h2{margin:0 0 18px;border-bottom:2px solid #374151;padding-bottom:10px}
h3{font-size:16px;margin:0 0 9px}.block{border:1px solid #d1d5db;border-radius:8px;padding:14px;margin:12px 0;background:#fff}
.block.task{background:#f9fafb;border-left:5px solid #374151}.text{line-height:1.75;white-space:normal;word-break:break-word}
.rating{margin-top:16px;padding:12px 14px;background:#fffbeb;border:1px solid #f59e0b;border-radius:8px;line-height:1.7}
@media print{body{background:#fff}.case{box-shadow:none;page-break-before:always}.intro{box-shadow:none}main{max-width:none;padding:0}}
</style></head><body><main>
<section class="intro"><h1>中医开放病例四系统专家盲审</h1>
<p>候选 A–D 的系统身份对每位专家、每道病例独立随机。请只依据专业质量与安全性判断，
不要尝试猜测来源。阅读稿不保存评分；请把选择填回配套 CSV/XLSX。</p>
<p>A/B/C/D=唯一最佳；TIE=并列；NONE=均不可接受；SKIP=无法评价并说明原因。</p></section>
""" + "".join(cards) + "</main></body></html>\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.input.expanduser().resolve(), args.output.expanduser().resolve())
    print(args.output)


if __name__ == "__main__":
    main()
