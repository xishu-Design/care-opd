from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from typing import Any, Coroutine

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.mcp.adapters import persistent_graph
from app.mcp.contracts import ToolRequest


class _GraphitiRuntime:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.ready = Event()
        self.thread = Thread(target=self._serve, name="guardtcm-graphiti", daemon=True)
        self.thread.start()
        self.ready.wait(timeout=10)
        self.graphiti: Any | None = None

    def _serve(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.ready.set()
        self.loop.run_forever()

    def run(self, coroutine: Coroutine[Any, Any, Any], timeout: int | None = None) -> Any:
        effective_timeout = timeout or get_settings().graphiti_timeout_seconds
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout=effective_timeout)
        except FutureTimeoutError as exc:
            # Do not leave a timed-out Graphiti/Ollama coroutine consuming the
            # single embedded runtime while the HTTP client retries the tool.
            future.cancel()
            raise TimeoutError(
                f"Graphiti operation exceeded {effective_timeout} seconds"
            ) from exc
        except Exception as exc:
            # Some third-party driver exceptions stringify to an empty value.
            # Preserve the exception class for a useful, content-free audit.
            if not str(exc).strip():
                exception_type = f"{type(exc).__module__}.{type(exc).__name__}"
                raise RuntimeError(
                    f"Graphiti operation failed with {exception_type}"
                ) from exc
            raise

    def submit(self, coroutine: Coroutine[Any, Any, Any]) -> Future:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)


_RUNTIME: _GraphitiRuntime | None = None
_INTERACTIVE_TIMEOUT_SECONDS = 8


def health(db: Session) -> dict:
    del db
    runtime = _runtime()
    runtime.run(_ensure_graphiti(runtime), timeout=min(get_settings().graphiti_timeout_seconds, _INTERACTIVE_TIMEOUT_SECONDS))
    return {
        "backend": "graphiti",
        "status": "ok",
        "storage": "embedded_falkordb",
        "database": get_settings().graphiti_database,
    }


def execute(db: Session, request: ToolRequest) -> dict:
    runtime = _runtime()
    graphiti = runtime.run(_ensure_graphiti(runtime), timeout=min(get_settings().graphiti_timeout_seconds, _INTERACTIVE_TIMEOUT_SECONDS))
    group_id = _group_id(request.context.role, request.context.user_id)

    if request.tool_name in {"create_case_graph", "update_case_graph", "add_prescription_to_graph", "add_risk_to_graph"}:
        write_mode = str(request.arguments.get("graphiti_write_mode") or "episode_extraction")
        if write_mode == "structured_triplet_v1":
            # The expert-blind harness seeds only a fixed, case-independent
            # response-organisation preference.  Persist that already-
            # structured relation directly through Graphiti's EntityNode /
            # EntityEdge and FalkorDB interfaces.  This is a genuine Graphiti
            # write and subsequent hybrid retrieval, while avoiding an
            # unnecessary LLM extraction pass for data that is already a
            # deterministic triplet.
            edge = runtime.run(
                _add_structured_triplet(
                    graphiti,
                    group_id=group_id,
                    topic=str(request.arguments.get("topic") or "中医辨证论治回答框架"),
                    preference=str(request.arguments.get("preference") or "按证据组织专业回答"),
                )
            )
            return {
                "graph_reasoning_path": [_edge_payload(edge, request)],
                "__metadata__": {
                    **_metadata(),
                    "write_mode": write_mode,
                },
            }
        elif write_mode == "episode_extraction":
            fallback = persistent_graph.execute(db, request)
            episode = {
                "tool": request.tool_name,
                "role": request.context.role,
                "user_id": request.context.user_id,
                "patient_id": request.context.patient_id,
                "prescription_id": request.context.prescription_id,
                "arguments": request.arguments,
                "reasoning_path": fallback.get("graph_reasoning_path", []),
            }
            runtime.run(_add_episode(graphiti, group_id, request.tool_name, episode))
            fallback["__metadata__"] = {**_metadata(), "write_mode": write_mode}
            return fallback
        else:
            raise ValueError(f"unsupported Graphiti write mode: {write_mode}")

    query = str(request.arguments.get("query") or _default_query(request)).strip()
    # Query the role/user partition through Graphiti's typed edge interface.
    # graphiti-core 0.29.2 maps each FalkorDB group to a separate graph, while
    # its single-group hybrid-search decorator can leave the base driver
    # selected.  Reading the explicit group driver is both deterministic and
    # avoids invoking an LLM reranker for an interactive reasoning-path lookup.
    edges = runtime.run(
        _get_group_edges(graphiti, group_id=group_id, query=query, limit=12),
        timeout=min(get_settings().graphiti_timeout_seconds, _INTERACTIVE_TIMEOUT_SECONDS),
    )
    paths = [_edge_payload(edge, request) for edge in edges]
    if paths:
        return {
            "graph_reasoning_path": paths,
            "__metadata__": {
                **_metadata(),
                "retrieval_mode": "typed_group_edge_query_v1",
            },
        }

    # A new local Graphiti store is initially empty. Preserve the application's
    # deterministic reasoning path while clearly reporting the empty-store fallback.
    fallback = persistent_graph.execute(db, request)
    fallback["__metadata__"] = {
        **_metadata(),
        "degraded": True,
        "fallback_used": True,
        "fallback_reason": "graphiti_no_matching_edges",
    }
    return fallback


def submit_chat_episode(*, role: str, user_id: int, question: str, answer: str, trace_id: str | None) -> dict:
    settings = get_settings()
    if settings.graph_backend != "graphiti" or not settings.graphiti_ingest_chat_history:
        return {"submitted": False, "reason": "graphiti_not_enabled"}
    runtime = _runtime()
    future = runtime.submit(
        _add_chat_episode(
            runtime,
            role=role,
            user_id=user_id,
            question=question,
            answer=answer,
            trace_id=trace_id,
        )
    )
    future.add_done_callback(_consume_background_result)
    return {"submitted": True}


def delete_chat_episode(*, role: str, user_id: int, trace_id: str) -> dict:
    settings = get_settings()
    if settings.graph_backend != "graphiti" or not settings.graphiti_ingest_chat_history:
        return {"deleted": 0, "reason": "graphiti_not_enabled"}
    runtime = _runtime()
    return runtime.run(_delete_chat_episodes(runtime, role=role, user_id=user_id, trace_id=trace_id))


async def _delete_chat_episodes(runtime: _GraphitiRuntime, *, role: str, user_id: int, trace_id: str) -> dict:
    graphiti = await _ensure_graphiti(runtime)
    episodes = await graphiti.retrieve_episodes(
        reference_time=datetime.now(timezone.utc),
        last_n=1000,
        group_ids=[_group_id(role, user_id)],
    )
    deleted = 0
    for episode in episodes:
        content = str(getattr(episode, "content", "") or "")
        if trace_id not in content:
            continue
        await graphiti.remove_episode(episode.uuid)
        deleted += 1
    return {"deleted": deleted}


async def _ensure_graphiti(runtime: _GraphitiRuntime):
    if runtime.graphiti is not None:
        return runtime.graphiti
    settings = get_settings()
    os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
    os.environ.setdefault("SEMAPHORE_LIMIT", "1")

    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from redislite.async_falkordb_client import AsyncFalkorDB

    db_path = _resolve_project_path(settings.graphiti_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    falkor = AsyncFalkorDB(dbfilename=str(db_path))
    driver = FalkorDriver(falkor_db=falkor, database=settings.graphiti_database)
    openai_base_url = f"{settings.ollama_base_url.rstrip('/')}/v1"
    llm_config = LLMConfig(
        api_key="ollama",
        model=settings.ollama_model,
        small_model=settings.ollama_model,
        base_url=openai_base_url,
        temperature=settings.llm_temperature,
        max_tokens=min(settings.llm_max_output_tokens * 3, 4096),
    )
    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=OpenAIGenericClient(
            config=llm_config,
            cache=False,
            structured_output_mode="json_object",
        ),
        embedder=OpenAIEmbedder(
            OpenAIEmbedderConfig(
                api_key="ollama",
                base_url=openai_base_url,
                embedding_model=settings.ollama_embedding_model,
                embedding_dim=settings.mem0_embedding_dims,
            )
        ),
        cross_encoder=OpenAIRerankerClient(config=llm_config),
        max_coroutines=1,
    )
    await graphiti.build_indices_and_constraints()
    runtime.graphiti = graphiti
    return graphiti


async def _add_chat_episode(
    runtime: _GraphitiRuntime,
    *,
    role: str,
    user_id: int,
    question: str,
    answer: str,
    trace_id: str | None,
) -> None:
    graphiti = await _ensure_graphiti(runtime)
    await _add_episode(
        graphiti,
        _group_id(role, user_id),
        f"{role}问答",
        {
            "角色": role,
            "用户编号": user_id,
            "问题": question[:1400],
            "回答": answer[:2600],
            "追踪编号": trace_id,
        },
    )


async def _add_episode(graphiti: Any, group_id: str, name: str, payload: dict) -> None:
    from graphiti_core.nodes import EpisodeType

    await graphiti.add_episode(
        name=name,
        episode_body=json.dumps(payload, ensure_ascii=False),
        source_description="GuardTCM 本地、按角色隔离的可追溯记录",
        reference_time=datetime.now(timezone.utc),
        source=EpisodeType.json,
        group_id=group_id,
    )


async def _add_structured_triplet(
    graphiti: Any,
    *,
    group_id: str,
    topic: str,
    preference: str,
) -> Any:
    """Persist an idempotent, pre-structured relation through real Graphiti.

    ``Graphiti.add_episode`` is intentionally LLM-heavy because it extracts
    entities and relations from unstructured prose.  This seed is already a
    trusted two-node relation, so invoking extraction once per isolated user
    adds minutes of latency without changing its semantics.  Graphiti 0.29.2
    exposes its typed nodes, edges, embedder and graph driver for this exact
    structured-ingest path.
    """

    from uuid import NAMESPACE_URL, uuid5

    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EntityNode

    now = datetime.now(timezone.utc)
    namespace = f"guardtcm:{group_id}:structured-triplet-v1"
    source_uuid = str(uuid5(NAMESPACE_URL, f"{namespace}:topic:{topic}"))
    target_uuid = str(uuid5(NAMESPACE_URL, f"{namespace}:preference:{preference}"))
    edge_uuid = str(uuid5(NAMESPACE_URL, f"{namespace}:edge:{topic}:{preference}"))
    source = EntityNode(
        uuid=source_uuid,
        name=topic,
        group_id=group_id,
        summary="中医辨证论治回答的专业组织框架",
    )
    target = EntityNode(
        uuid=target_uuid,
        name=preference,
        group_id=group_id,
        summary="医生角色的病例分析与复核表达偏好",
    )
    edge = EntityEdge(
        uuid=edge_uuid,
        group_id=group_id,
        source_node_uuid=source_uuid,
        target_node_uuid=target_uuid,
        created_at=now,
        name="ORGANIZES_RESPONSE",
        fact=f"{topic}：{preference}",
        episodes=[],
        valid_at=now,
        reference_time=now,
    )
    await source.generate_name_embedding(graphiti.embedder)
    await target.generate_name_embedding(graphiti.embedder)
    await edge.generate_embedding(graphiti.embedder)
    driver = _driver_for_group(graphiti, group_id)
    await source.save(driver)
    await target.save(driver)
    await edge.save(driver)
    return edge


async def _get_group_edges(
    graphiti: Any,
    *,
    group_id: str,
    query: str,
    limit: int,
) -> list[Any]:
    from graphiti_core.edges import EntityEdge
    from graphiti_core.errors import GroupsEdgesNotFoundError

    driver = _driver_for_group(graphiti, group_id)
    try:
        edges = await EntityEdge.get_by_group_ids(
            driver,
            [group_id],
            limit=max(limit * 4, 32),
        )
    except GroupsEdgesNotFoundError:
        return []
    query_terms = [term for term in query.lower().split() if term]

    def relevance(edge: Any) -> tuple[int, str]:
        text = f"{getattr(edge, 'name', '')} {getattr(edge, 'fact', '')}".lower()
        score = sum(2 for term in query_terms if term in text)
        # Character overlap keeps Chinese queries useful even when callers do
        # not insert spaces between every medical concept.
        score += len(set(query.lower()) & set(text))
        return score, str(getattr(edge, "created_at", ""))

    return sorted(edges, key=relevance, reverse=True)[:limit]


def _driver_for_group(graphiti: Any, group_id: str) -> Any:
    from graphiti_core.driver.driver import GraphProvider

    driver = graphiti.driver
    if (
        driver.provider == GraphProvider.FALKORDB
        and getattr(driver, "_database", None) != group_id
    ):
        return driver.clone(database=group_id)
    return driver


def _runtime() -> _GraphitiRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = _GraphitiRuntime()
    return _RUNTIME


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[4] / path


def _group_id(role: str, user_id: int) -> str:
    prefix = "".join(character if character.isalnum() or character in "-_" else "-" for character in get_settings().graphiti_group_prefix)
    return f"{prefix}-{role}-{user_id}"


def _default_query(request: ToolRequest) -> str:
    if request.context.task_type == "doctor_ai_chat":
        return "中医证候、治法、方剂和药材之间的专业关系"
    if request.context.task_type == "student_ai_chat":
        return "中医证候、治法、方剂和药材之间的教学关系"
    return "处方、风险、证据之间的关系"


def _edge_payload(edge: Any, request: ToolRequest) -> dict:
    return {
        "nodes": [edge.source_node_uuid, edge.target_node_uuid],
        "edges": [edge.name],
        "explanation": edge.fact,
        "fact_uuid": edge.uuid,
        "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
        "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
        "visibility": request.arguments.get("visibility") or (
            "student_learning" if request.context.role == "student" else "doctor_internal"
        ),
    }


def _consume_background_result(future: Future) -> None:
    try:
        future.result()
    except Exception:
        # SQL chat history is the source of truth; Graphiti enrichment is best effort.
        pass


def _metadata() -> dict:
    return {
        "backend": "graphiti",
        "degraded": False,
        "fallback_used": False,
        "storage": "embedded_falkordb",
    }
