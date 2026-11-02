"""Exact and character-ngram overlap audit for external domain benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SPACE = re.compile(r"\s+")
PUNCT = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)


def normalize(text: str) -> str:
    return PUNCT.sub("", SPACE.sub("", text).lower())


def char_ngrams(text: str, n: int) -> set[str]:
    value = normalize(text)
    if len(value) < n:
        return {value} if value else set()
    return {value[index : index + n] for index in range(len(value) - n + 1)}


def _extract(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    parts: list[str] = []
    for field in fields:
        value = row.get(field)
        if isinstance(value, str):
            parts.append(value)
        elif value is not None:
            parts.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return "\n".join(parts)


def _read(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def audit(
    train_path: Path,
    eval_path: Path,
    *,
    train_fields: Sequence[str],
    eval_fields: Sequence[str],
    ngram: int = 13,
    threshold: float = 0.5,
) -> dict[str, Any]:
    train_hashes: set[str] = set()
    inverted: dict[str, set[int]] = defaultdict(set)
    train_texts: list[str] = []
    for _, row in _read(train_path):
        text = normalize(_extract(row, train_fields))
        if not text:
            continue
        index = len(train_texts)
        train_texts.append(text)
        train_hashes.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
        for gram in char_ngrams(text, ngram):
            inverted[gram].add(index)

    exact: list[dict[str, Any]] = []
    near: list[dict[str, Any]] = []
    eval_count = 0
    for line_number, row in _read(eval_path):
        text = normalize(_extract(row, eval_fields))
        if not text:
            continue
        eval_count += 1
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in train_hashes:
            exact.append({"eval_line": line_number, "sha256": digest})
            continue
        grams = char_ngrams(text, ngram)
        candidates: set[int] = set()
        for gram in grams:
            candidates.update(inverted.get(gram, ()))
        best = (0.0, -1)
        for index in candidates:
            train_grams = char_ngrams(train_texts[index], ngram)
            union = grams | train_grams
            score = len(grams & train_grams) / len(union) if union else 0.0
            if score > best[0]:
                best = (score, index)
        if best[0] >= threshold:
            near.append(
                {
                    "eval_line": line_number,
                    "train_index": best[1],
                    "jaccard": best[0],
                }
            )
    return {
        "train_records": len(train_texts),
        "eval_records": eval_count,
        "ngram": ngram,
        "threshold": threshold,
        "exact_matches": exact,
        "near_matches": near,
        "passed": not exact and not near,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--train-fields", default="prompt,target")
    parser.add_argument("--eval-fields", default="question,analysis")
    parser.add_argument("--ngram", type=int, default=13)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.train,
        args.eval,
        train_fields=args.train_fields.split(","),
        eval_fields=args.eval_fields.split(","),
        ngram=args.ngram,
        threshold=args.threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

