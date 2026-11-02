from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import AdapterHealthCheck
from app.mcp.adapters import (
    auth_mock,
    graphiti_real,
    graphiti_mock,
    lightrag_adapter,
    learning_mock,
    lightrag_mock,
    local_knowledge,
    mem0_mock,
    mem0_real,
    patient_record_mock,
    persistent_graph,
    persistent_memory,
    prescription_mock,
    rule_engine_mock,
    rule_engine_real,
)
from app.mcp.contracts import ToolRequest, ToolSpec

AdapterExecute = Callable[[Session, ToolRequest], dict]


@dataclass
class AdapterSelection:
    execute: AdapterExecute
    backend: str
    configured_backend: str
    fallback_used: bool = False
    fallback_reason: str | None = None
    status: str = "ok"


class AdapterRegistry:
    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()

    def execute(self, spec: ToolSpec, request: ToolRequest) -> tuple[dict, dict]:
        selection = self.select(spec)
        data = selection.execute(self.db, request)
        metadata = data.pop("__metadata__", {}) if isinstance(data, dict) else {}
        base_metadata = {
            "backend": metadata.get("backend", selection.backend),
            "configured_backend": selection.configured_backend,
            "degraded": bool(metadata.get("degraded", selection.fallback_used)),
            "fallback_used": selection.fallback_used or bool(metadata.get("fallback_used", False)),
            "fallback_reason": selection.fallback_reason or metadata.get("fallback_reason"),
            "adapter_version": "0.3.0",
        }
        for key, value in metadata.items():
            base_metadata.setdefault(key, value)
        if metadata.get("degraded_reason"):
            base_metadata["degraded_reason"] = metadata["degraded_reason"]
        return data, base_metadata

    def select(self, spec: ToolSpec) -> AdapterSelection:
        if spec.server == "KnowledgeServer":
            return self._select_knowledge()
        if spec.server == "RuleEngineServer":
            return self._select_configured(
                configured=self.settings.rule_engine_backend,
                real_backend="local_rules",
                real_execute=rule_engine_real.execute,
                real_health=rule_engine_real.health,
                fallback_execute=rule_engine_mock.execute,
                fallback_backend="mock_rule_engine",
            )
        if spec.server == "GraphServer":
            return self._select_graph()
        if spec.server == "MemoryServer":
            return self._select_memory()
        return self._select_mock_server(spec.server)

    def _select_knowledge(self) -> AdapterSelection:
        configured = self.settings.knowledge_backend
        if configured == "lightrag":
            try:
                health = lightrag_adapter.health(self.db)
                status = health.get("status", "unavailable")
                self._record_health("lightrag", configured, status, health)
                if status == "ok":
                    return AdapterSelection(lightrag_adapter.execute, "lightrag", configured, status=status)
                return self._fallback(configured, local_knowledge.execute, "local_knowledge", f"lightrag_health_{status}")
            except Exception as exc:
                self._record_health("lightrag", configured, "error", {"error": str(exc)})
                return self._fallback(configured, local_knowledge.execute, "local_knowledge", "lightrag_health_error")
        if configured == "local_knowledge":
            return self._select_configured(
                configured=configured,
                real_backend="local_knowledge",
                real_execute=local_knowledge.execute,
                real_health=local_knowledge.health,
                fallback_execute=lightrag_mock.execute,
                fallback_backend="mock_lightrag",
            )
        if configured == "mock":
            return AdapterSelection(lightrag_mock.execute, "mock_lightrag", configured)
        return self._fallback(configured, local_knowledge.execute, "local_knowledge", f"unsupported_backend:{configured}")

    def _select_graph(self) -> AdapterSelection:
        configured = self.settings.graph_backend
        if configured == "graphiti":
            return self._select_configured(
                configured=configured,
                real_backend="graphiti",
                real_execute=graphiti_real.execute,
                real_health=graphiti_real.health,
                fallback_execute=persistent_graph.execute,
                fallback_backend="persistent_graph",
            )
        if configured == "persistent_graph":
            return self._select_configured(
                configured=configured,
                real_backend="persistent_graph",
                real_execute=persistent_graph.execute,
                real_health=persistent_graph.health,
                fallback_execute=graphiti_mock.execute,
                fallback_backend="mock_graphiti",
            )
        if configured == "mock":
            return AdapterSelection(graphiti_mock.execute, "mock_graphiti", configured)
        return self._fallback(configured, persistent_graph.execute, "persistent_graph", f"unsupported_backend:{configured}")

    def _select_memory(self) -> AdapterSelection:
        configured = self.settings.memory_backend
        if configured == "mem0":
            return self._select_configured(
                configured=configured,
                real_backend="mem0",
                real_execute=mem0_real.execute,
                real_health=mem0_real.health,
                fallback_execute=persistent_memory.execute,
                fallback_backend="persistent_memory",
            )
        if configured == "persistent_memory":
            return self._select_configured(
                configured=configured,
                real_backend="persistent_memory",
                real_execute=persistent_memory.execute,
                real_health=persistent_memory.health,
                fallback_execute=mem0_mock.execute,
                fallback_backend="mock_mem0",
            )
        if configured == "mock":
            return AdapterSelection(mem0_mock.execute, "mock_mem0", configured)
        return self._fallback(configured, persistent_memory.execute, "persistent_memory", f"unsupported_backend:{configured}")

    def tool_status(self, spec: ToolSpec) -> dict:
        selection = self.select(spec)
        return {
            "backend": selection.backend,
            "configured_backend": selection.configured_backend,
            "status": selection.status,
            "fallback_used": selection.fallback_used,
            "fallback_reason": selection.fallback_reason,
        }

    def _select_configured(
        self,
        *,
        configured: str,
        real_backend: str,
        real_execute: AdapterExecute,
        real_health: Callable[[Session], dict],
        fallback_execute: AdapterExecute,
        fallback_backend: str,
        unavailable_backends: set[str] | None = None,
    ) -> AdapterSelection:
        unavailable_backends = unavailable_backends or set()
        if configured == real_backend:
            try:
                health = real_health(self.db)
                status = health.get("status", "ok")
                self._record_health(real_backend, real_backend, status, health)
                if status == "ok":
                    return AdapterSelection(real_execute, real_backend, configured, status=status)
                return self._fallback(configured, fallback_execute, fallback_backend, f"{real_backend}_health_{status}")
            except Exception as exc:
                self._record_health(real_backend, real_backend, "error", {"error": str(exc)})
                return self._fallback(configured, fallback_execute, fallback_backend, f"{real_backend}_health_error")

        if configured == "mock":
            return AdapterSelection(fallback_execute, fallback_backend, configured)

        if configured in unavailable_backends:
            return self._fallback(configured, fallback_execute, fallback_backend, f"{configured}_not_configured")

        return self._fallback(configured, fallback_execute, fallback_backend, f"unsupported_backend:{configured}")

    def _fallback(
        self,
        configured: str,
        execute: AdapterExecute,
        backend: str,
        reason: str,
    ) -> AdapterSelection:
        if not self.settings.allow_adapter_fallback:
            raise RuntimeError(f"Adapter backend unavailable and fallback disabled: {reason}")
        self._record_health(backend, configured, "fallback", {"reason": reason})
        return AdapterSelection(
            execute=execute,
            backend=backend,
            configured_backend=configured,
            fallback_used=True,
            fallback_reason=reason,
            status="fallback",
        )

    def _select_mock_server(self, server: str) -> AdapterSelection:
        execute = {
            "AuthMockServer": auth_mock.execute,
            "PatientRecordMockServer": patient_record_mock.execute,
            "PrescriptionMockServer": prescription_mock.execute,
            "RuleEngineMockServer": rule_engine_mock.execute,
            "LightRAGMockServer": lightrag_mock.execute,
            "GraphitiMockServer": graphiti_mock.execute,
            "Mem0MockServer": mem0_mock.execute,
            "LearningMockServer": learning_mock.execute,
        }[server]
        backend = {
            "AuthMockServer": "mock_auth",
            "PatientRecordMockServer": "mock_patient_record",
            "PrescriptionMockServer": "mock_prescription",
            "RuleEngineMockServer": "mock_rule_engine",
            "LightRAGMockServer": "mock_lightrag",
            "GraphitiMockServer": "mock_graphiti",
            "Mem0MockServer": "mock_mem0",
            "LearningMockServer": "mock_learning",
        }[server]
        return AdapterSelection(execute, backend, backend)

    def _record_health(self, adapter_name: str, backend: str, status: str, details: dict) -> None:
        self.db.add(
            AdapterHealthCheck(
                adapter_name=adapter_name,
                backend=backend,
                status=status,
                details_json=details,
            )
        )
        self.db.flush()
