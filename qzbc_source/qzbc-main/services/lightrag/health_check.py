from __future__ import annotations

import argparse
import json
import os
import urllib.request


REQUIRED_PATHS = {
    "/health",
    "/documents/texts",
    "/documents/track_status/{track_id}",
    "/query",
    "/query/data",
    "/graph/label/search",
    "/graphs",
}


def check(base_url: str, api_key: str) -> dict[str, object]:
    headers = {"X-API-Key": api_key} if api_key else {}
    health = _get_json(f"{base_url.rstrip('/')}/health", headers)
    openapi = _get_json(f"{base_url.rstrip('/')}/openapi.json", headers)
    paths = set(openapi.get("paths", {})) if isinstance(openapi, dict) else set()
    missing = sorted(REQUIRED_PATHS - paths)
    return {
        "status": "ok" if not missing else "incompatible",
        "health": health,
        "required_paths": sorted(REQUIRED_PATHS),
        "missing_paths": missing,
    }


def _get_json(url: str, headers: dict[str, str]) -> dict[str, object]:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8") or "{}")
    if not isinstance(payload, dict):
        raise ValueError(f"{url} 未返回 JSON 对象")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 LightRAG 健康状态与 OpenAPI 契约。")
    parser.add_argument("--base-url", default=os.getenv("LIGHTRAG_API_BASE_URL", "http://127.0.0.1:9621"))
    parser.add_argument("--api-key", default=os.getenv("LIGHTRAG_API_KEY", ""))
    args = parser.parse_args()
    result = check(args.base_url, args.api_key)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
