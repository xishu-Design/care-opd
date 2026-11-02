from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def _json(url: str, headers: dict[str, str] | None = None) -> dict:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=3) as response:
        payload = json.loads(response.read().decode("utf-8") or "{}")
    return payload if isinstance(payload, dict) else {}


def main() -> int:
    result = {
        "ollama": "unavailable",
        "generation_model": "missing",
        "embedding_model": "missing",
        "lightrag": "unavailable",
        "backend": "unavailable",
        "frontend": "unavailable",
        "ready": False,
    }
    try:
        tags = _json("http://127.0.0.1:11434/api/tags")
        names = {str(item.get("name", "")) for item in tags.get("models", [])}
        result["ollama"] = "ok"
        result["generation_model"] = "ok" if "qwen2.5:7b" in names else "missing"
        result["embedding_model"] = "ok" if {"bge-m3:latest", "bge-m3"} & names else "missing"
    except Exception:
        pass
    try:
        key = os.getenv("LIGHTRAG_API_KEY", "")
        _json("http://127.0.0.1:9621/health", {"X-API-Key": key} if key else {})
        result["lightrag"] = "ok"
    except Exception:
        pass
    for name, url in (("backend", "http://127.0.0.1:8000/health"), ("frontend", "http://127.0.0.1:5173")):
        try:
            urllib.request.urlopen(url, timeout=2).close()
            result[name] = "ok"
        except Exception:
            pass
    result["ready"] = all(result[key] == "ok" for key in ("ollama", "generation_model", "embedding_model", "lightrag", "backend", "frontend"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
