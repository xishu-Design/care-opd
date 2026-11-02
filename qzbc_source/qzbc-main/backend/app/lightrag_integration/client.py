from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from app.lightrag_integration.config import LightRAGConfig, get_lightrag_config
from app.lightrag_integration.document_builder import load_lightrag_docs
from app.lightrag_integration.schemas import LightRAGDoc, LightRAGQueryResponse, LightRAGQueryResult


class LightRAGClient:
    def __init__(self, config: LightRAGConfig | None = None) -> None:
        self.config = config or get_lightrag_config()

    def health(self) -> dict[str, Any]:
        if not self.config.enabled:
            return self._health("unavailable", "lightrag_disabled")
        if self.config.mode == "api":
            return self._api_health()
        docs_count = len(self._embedded_docs())
        status = "ok" if docs_count else "degraded"
        return {
            "enabled": True,
            "backend": "lightrag",
            "mode": self.config.mode,
            "status": status,
            "working_dir": str(self.config.working_dir),
            "docs_count": docs_count,
            "implementation": "embedded_document_store",
            "degraded": docs_count == 0,
            "degraded_reason": "" if docs_count else "no_lightrag_docs_found",
        }

    def query(self, text: str, mode: str = "hybrid", top_k: int = 5) -> LightRAGQueryResponse:
        started = time.perf_counter()
        if not self.config.enabled:
            return self._failure(text, mode, started, "lightrag_disabled")
        if self.config.mode == "api":
            return self._api_query(text, mode, top_k, started)
        docs = self._embedded_docs()
        results = _search_docs(docs, text, top_k)
        return LightRAGQueryResponse(
            success=True,
            query=text,
            results=results,
            metadata={
                "backend": "lightrag",
                "mode": mode,
                "lightrag_mode": self.config.mode,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "implementation": "embedded_document_store",
                "degraded": False,
            },
        )

    def _api_health(self) -> dict[str, Any]:
        try:
            payload = self._request_json("GET", "/health")
        except Exception as exc:
            return self._health("unavailable", str(exc))
        raw_status = str(payload.get("status", "ok")) if isinstance(payload, dict) else "ok"
        status = "ok" if raw_status.lower() in {"ok", "healthy", "ready", "success"} else raw_status
        return {
            "enabled": True,
            "backend": "lightrag",
            "mode": "api",
            "status": status,
            "api_base_url": self.config.api_base_url,
            "working_dir": str(self.config.working_dir),
            "docs_count": 0,
            "raw_status": raw_status,
            "raw": payload,
        }

    def _api_query(self, text: str, mode: str, top_k: int, started: float) -> LightRAGQueryResponse:
        try:
            payload = self.query_data(text, mode=mode, top_k=top_k)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return self._failure(text, mode, started, str(exc))
        results = [_result_from_raw(item) for item in _raw_results(payload, self._embedded_docs())]
        return LightRAGQueryResponse(
            success=True,
            query=text,
            results=results,
            metadata={
                "backend": "lightrag",
                "mode": mode,
                "lightrag_mode": "api",
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "raw_status": payload.get("status") if isinstance(payload, dict) else None,
            },
        )

    def openapi_compatibility(self) -> dict[str, Any]:
        required = {
            "/health",
            "/documents/texts",
            "/documents/track_status/{track_id}",
            "/query",
            "/query/data",
            "/graph/label/search",
            "/graphs",
        }
        try:
            payload = self._request_json("GET", "/openapi.json")
            paths = set(payload.get("paths", {})) if isinstance(payload, dict) else set()
            missing = sorted(required - paths)
            return {"compatible": not missing, "missing_paths": missing, "available_paths": sorted(paths)}
        except Exception as exc:
            return {"compatible": False, "missing_paths": sorted(required), "error": str(exc)}

    def query_data(self, text: str, *, mode: str = "mix", top_k: int = 10) -> dict[str, Any]:
        payload = self._request_json("POST", "/query/data", {"query": _api_query_text(text), "mode": mode, "top_k": top_k})
        return payload if isinstance(payload, dict) else {"data": payload}

    def graph_labels(self, query: str = "", limit: int = 50) -> Any:
        suffix = urllib.parse.urlencode({"q": query, "limit": max(1, min(limit, 50))})
        return self._request_json("GET", f"/graph/label/search?{suffix}")

    def graph(self, label: str, max_depth: int = 2, max_nodes: int = 200) -> Any:
        if not self.config.enabled:
            raise RuntimeError("lightrag_disabled")
        suffix = urllib.parse.urlencode({"label": label, "max_depth": max_depth, "max_nodes": max_nodes})
        # Graph browsing must stay responsive even when the optional external
        # LightRAG server is unhealthy. The API route falls back to the local
        # reviewed graph when this bounded request fails.
        return self._request_json("GET", f"/graphs?{suffix}", timeout_seconds=min(self.config.timeout_seconds, 8))

    def upload_texts(self, texts: list[str], file_sources: list[str] | None = None) -> Any:
        body: dict[str, Any] = {"texts": texts}
        if file_sources:
            body["file_sources"] = file_sources
        return self._request_json("POST", "/documents/texts", body)

    def track_status(self, track_id: str) -> Any:
        return self._request_json("GET", f"/documents/track_status/{urllib.parse.quote(track_id, safe='')}")

    def _request_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
    ) -> Any:
        headers = {"Accept": "application/json"}
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(f"{self.config.api_base_url}{path}", data=data, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=timeout_seconds or self.config.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw or "{}")

    def _embedded_docs(self) -> list[LightRAGDoc]:
        docs_path = self.config.index_path if self.config.index_path.exists() else self.config.input_docs_path
        return load_lightrag_docs(docs_path)

    def _failure(self, text: str, mode: str, started: float, error: str) -> LightRAGQueryResponse:
        return LightRAGQueryResponse(
            success=False,
            query=text,
            results=[],
            error=error,
            metadata={
                "backend": "lightrag",
                "mode": mode,
                "lightrag_mode": self.config.mode,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "degraded": True,
                "degraded_reason": "lightrag_query_failed",
            },
        )

    def _health(self, status: str, reason: str) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "backend": "lightrag",
            "mode": self.config.mode,
            "status": status,
            "working_dir": str(self.config.working_dir),
            "docs_count": len(self._embedded_docs()) if self.config.input_docs_path.exists() or self.config.index_path.exists() else 0,
            "degraded": status != "ok",
            "degraded_reason": reason,
        }


def track_state(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    direct = str(payload.get("status") or payload.get("track_status") or "").lower()
    if direct:
        return direct
    documents = payload.get("documents")
    if isinstance(documents, list) and documents:
        states = {str(item.get("status") or "").lower() for item in documents if isinstance(item, dict)}
        if states & {"failed", "error", "cancelled", "canceled"}:
            return "failed"
        if states and states <= {"processed", "completed", "complete", "done", "success"}:
            return "processed"
        return "processing"
    summary = payload.get("status_summary")
    if isinstance(summary, dict) and summary:
        states = {str(key).split(".")[-1].lower() for key, count in summary.items() if count}
        if states & {"failed", "error", "cancelled", "canceled"}:
            return "failed"
        if states and states <= {"processed", "completed", "complete", "done", "success"}:
            return "processed"
        return "processing"
    return ""


def _api_query_text(text: str) -> str:
    query = text.strip()
    return query if len(query) >= 3 else f"{query} 中药"


def _search_docs(docs: list[LightRAGDoc], query: str, top_k: int) -> list[LightRAGQueryResult]:
    scored = []
    for doc in docs:
        score = _score(doc, query)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [_result_from_doc(doc, min(score, 1.0)) for score, doc in scored[:top_k]]


def _score(doc: LightRAGDoc, query: str) -> float:
    query = (query or "").strip().lower()
    if not query:
        return 0.0
    haystack = json.dumps(doc.model_dump(mode="json"), ensure_ascii=False).lower()
    score = 0.0
    if query in doc.title.lower():
        score += 0.6
    if query in haystack:
        score += 0.4
    for token in _tokens(query):
        if token in haystack:
            score += 0.12
    return score


def _tokens(query: str) -> list[str]:
    text = query
    for separator in (" ", "，", ",", "。", "；", ";", "、"):
        text = text.replace(separator, " ")
    return [token for token in text.split(" ") if token]


def _result_from_doc(doc: LightRAGDoc, score: float) -> LightRAGQueryResult:
    source_id = str(doc.metadata.get("source_id", ""))
    graph_entities = [
        {"id": doc.entry_id, "name": doc.title, "type": doc.entry_type},
        {"id": source_id, "name": doc.metadata.get("source_title") or source_id, "type": "source"},
    ]
    graph_relations = [{"source": doc.entry_id, "target": source_id, "type": "derived_from"}] if source_id else []
    return LightRAGQueryResult(
        content=doc.text,
        source_id=source_id,
        entry_id=doc.entry_id,
        entry_type=doc.entry_type,
        title=doc.title,
        score=round(max(score, 0.01), 2),
        graph_entities=graph_entities,
        graph_relations=graph_relations,
        graph_path=graph_relations,
        metadata=doc.metadata,
        raw=doc.model_dump(mode="json"),
    )


def _raw_results(payload: Any, docs: list[LightRAGDoc] | None = None) -> list[Any]:
    if isinstance(payload, dict):
        results = payload.get("results") or payload.get("data") or []
        if isinstance(results, list):
            return results
        if isinstance(results, str):
            return [{"content": results}]
        if isinstance(results, dict) and docs:
            return _structured_query_results(results, docs)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, str):
        return [{"content": payload}]
    return []


def _structured_query_results(data: dict[str, Any], docs: list[LightRAGDoc]) -> list[dict[str, Any]]:
    doc_map = {doc.doc_id: doc for doc in docs}
    entities = data.get("entities") if isinstance(data.get("entities"), list) else []
    relationships = data.get("relationships") if isinstance(data.get("relationships"), list) else []
    chunks = data.get("chunks") if isinstance(data.get("chunks"), list) else []
    references = data.get("references") if isinstance(data.get("references"), list) else []
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id") or "")
        for doc_id in _file_doc_ids(chunk.get("file_path")):
            doc = doc_map.get(doc_id)
            if doc is None or (doc.doc_id, chunk_id) in seen:
                continue
            seen.add((doc.doc_id, chunk_id))
            results.append(
                {
                    "content": str(chunk.get("content") or doc.text),
                    "source_id": str(doc.metadata.get("source_id") or ""),
                    "entry_id": doc.entry_id,
                    "entry_type": doc.entry_type,
                    "title": doc.title,
                    "score": 1.0,
                    "graph_entities": [item for item in entities if doc.doc_id in _file_doc_ids(item.get("file_path"))],
                    "graph_relations": [item for item in relationships if doc.doc_id in _file_doc_ids(item.get("file_path"))],
                    "graph_path": [],
                    "metadata": {**doc.metadata, "chunk_id": chunk_id, "references": references},
                }
            )
    return results


def _file_doc_ids(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split("<SEP>") if part.strip().startswith("DOC_")]


def _result_from_raw(item: Any) -> LightRAGQueryResult:
    raw = item if isinstance(item, dict) else {"content": str(item)}
    metadata = raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {}
    content = str(raw.get("content") or raw.get("text") or raw.get("snippet") or "")
    return LightRAGQueryResult(
        content=content,
        source_id=str(raw.get("source_id") or metadata.get("source_id") or ""),
        entry_id=str(raw.get("entry_id") or metadata.get("entry_id") or ""),
        entry_type=str(raw.get("entry_type") or metadata.get("entry_type") or ""),
        title=str(raw.get("title") or metadata.get("title") or ""),
        score=float(raw.get("score") or raw.get("confidence") or 0.5),
        graph_entities=raw.get("graph_entities", []),
        graph_relations=raw.get("graph_relations", []),
        graph_path=raw.get("graph_path", []),
        metadata=metadata,
        raw=raw,
    )
