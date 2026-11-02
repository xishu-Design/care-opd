"""Create Chinese display fields for catalog data with a local Ollama model.

Original source fields are retained in ``original_name`` and
``original_functions``.  This is a display translation only: it neither adds
medical claims nor changes the source provenance.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from app.content.validators.content_validator import DEFAULT_CONTENT_DATA_DIR


MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
HAN_RE = re.compile(r"[\u3400-\u9fff]")


def localize_catalog(*, content_dir: Path | str = DEFAULT_CONTENT_DATA_DIR, batch_size: int = 120, include_formulas: bool = False, max_batches: int | None = None, names_only: bool = False) -> dict[str, int]:
    root = Path(content_dir)
    herbs = _items(root / "herbs.json")
    formulas = _items(root / "formulas.json")
    targets = [*herbs, *formulas] if include_formulas else herbs
    cache_path = root / ".localization_zh_cache.json"
    cache = _cache(cache_path)
    names = _translation_map(_pending_names(targets), kind="名称", batch_size=batch_size, cache=cache, cache_key="names", cache_path=cache_path, max_batches=max_batches)
    functions = cache.get("functions", {}) if names_only else _translation_map(_pending_functions(targets), kind="中药功效", batch_size=batch_size, cache=cache, cache_key="functions", cache_path=cache_path, max_batches=max_batches)
    _apply(herbs, names, functions)
    if include_formulas:
        _apply(formulas, names, functions)
    _write(root / "herbs.json", {"items": herbs})
    _write(root / "formulas.json", {"items": formulas})
    return {"names_translated": len(names), "functions_translated": len(functions), "herbs": len(herbs), "formulas": len(formulas)}


def _pending_names(items: list[dict[str, Any]]) -> list[str]:
    return sorted({str(item.get("name", "")).strip() for item in items if item.get("name") and not HAN_RE.search(str(item["name"]))})


def _pending_functions(items: list[dict[str, Any]]) -> list[str]:
    return sorted({str(value).strip() for item in items for value in item.get("functions", []) if value and not HAN_RE.search(str(value))})


def _translation_map(values: list[str], *, kind: str, batch_size: int, cache: dict[str, dict[str, str]], cache_key: str, cache_path: Path, max_batches: int | None) -> dict[str, str]:
    translated = {source: target for source, target in cache.get(cache_key, {}).items() if source in values and HAN_RE.search(target)}
    pending = [value for value in values if value not in translated]
    for index, start in enumerate(range(0, len(pending), batch_size), start=1):
        if max_batches is not None and index > max_batches:
            break
        batch = pending[start : start + batch_size]
        response = _translate_batch(batch, kind)
        missing = set(batch) - set(response)
        if missing:
            # A long JSON response can occasionally omit a tail item. Retry
            # those entries separately rather than accepting partial Chinese UI.
            response.update(_translate_batch(sorted(missing), kind))
            missing = set(batch) - set(response)
        invalid = {source: target for source, target in response.items() if not HAN_RE.search(target)}
        if invalid:
            response.update(_translate_batch(sorted(invalid), kind))
        translated.update({source: target for source, target in response.items() if source in batch and HAN_RE.search(target)})
        cache[cache_key] = translated
        _write(cache_path, cache)
        print(f"translated {kind}: {len(translated)}/{len(values)}", flush=True)
    return translated


def _translate_batch(values: list[str], kind: str) -> dict[str, str]:
    prompt = (
        "你是中医术语翻译员。把下列每一个英文或拼音术语译为简洁、规范的中文。"
        "这是目录显示翻译，不得添加疗效、剂量、适应症或解释；名称尽量使用标准中药名。"
        "只输出与输入完全同序的行，格式必须为“序号<TAB>中文翻译”，不要 Markdown、解释或其他文字。\n"
        f"类别：{kind}\n输入：\n" + "\n".join(f"{index}\t{value}" for index, value in enumerate(values, start=1))
    )
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0, "num_predict": 8192}}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as reply:
        payload = json.loads(reply.read().decode())
    result = str(payload["response"])
    translated: dict[str, str] = {}
    for line in result.splitlines():
        match = re.match(r"^\s*(\d+)(?:\s+|[|:：.]|、)\s*(.*?)\s*$", line)
        if match and match.group(2):
            index = int(match.group(1)) - 1
            if 0 <= index < len(values):
                translated[values[index]] = match.group(2)
    return translated


def _apply(items: list[dict[str, Any]], names: dict[str, str], functions: dict[str, str]) -> None:
    for item in items:
        name = str(item.get("name", ""))
        if name in names:
            item.setdefault("original_name", name)
            item["display_name"] = names[name]
        raw_functions = [str(value) for value in item.get("functions", []) if value]
        display_functions = [functions.get(value, value) for value in raw_functions]
        if display_functions and all(HAN_RE.search(value) for value in display_functions):
            item.setdefault("original_functions", raw_functions)
            item["display_functions"] = display_functions
        item["translation_note"] = "中文展示字段由本地 qwen2.5:7b 翻译生成；原始名称与功效字段已保留，需人工复核。"


def _items(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")).get("items", [])


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {"names": {}, "functions": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "names": {str(key): str(value) for key, value in data.get("names", {}).items()},
        "functions": {str(key): str(value) for key, value in data.get("functions", {}).items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Chinese display translations with local Ollama.")
    parser.add_argument("--content-dir", default=str(DEFAULT_CONTENT_DATA_DIR))
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--include-formulas", action="store_true", help="Also translate formula titles and functions (disabled by default to exclude numbered condition labels).")
    parser.add_argument("--max-batches", type=int, help="Translate at most this many batches of each type; safe to rerun.")
    parser.add_argument("--names-only", action="store_true", help="Translate names only; useful when resuming a large formula catalog.")
    args = parser.parse_args()
    print(json.dumps(localize_catalog(content_dir=args.content_dir, batch_size=args.batch_size, include_formulas=args.include_formulas, max_batches=args.max_batches, names_only=args.names_only), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
