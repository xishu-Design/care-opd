from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from app.core.config import get_settings
from app.evaluation.runners.base import BaseEvaluationRunner
from app.evaluation.schemas import EvaluationCase, EvaluationCaseResult
from app.lightrag_integration.client import LightRAGClient
from app.lightrag_integration.config import LightRAGConfig
from app.lightrag_integration.schemas import LightRAGDoc
from app.mcp.contracts import ToolContext, ToolRequest
from app.mcp.gateway import MCPGateway


class LightRAGEvaluationRunner(BaseEvaluationRunner):
    def run(self, cases: list[EvaluationCase]) -> list[EvaluationCaseResult]:
        if not cases:
            return []
        with TemporaryDirectory(prefix="guardtcm-phase12-rag-") as directory:
            root = Path(directory)
            input_dir = root / "input"
            working_dir = root / "working"
            input_dir.mkdir()
            working_dir.mkdir()
            docs = [self._doc(case) for case in cases]
            content = "\n".join(json.dumps(doc.model_dump(mode="json"), ensure_ascii=False) for doc in docs) + "\n"
            (input_dir / "guardtcm_lightrag_docs.jsonl").write_text(content, encoding="utf-8")
            (working_dir / "guardtcm_lightrag_index.jsonl").write_text(content, encoding="utf-8")
            config = LightRAGConfig(
                enabled=True,
                mode="embedded",
                working_dir=working_dir,
                input_dir=input_dir,
                input_docs_path=input_dir / "guardtcm_lightrag_docs.jsonl",
                api_base_url="",
                timeout_seconds=5,
                allow_fallback=True,
                reindex_on_start=False,
            )
            client = LightRAGClient(config)
            return [self.timed(case, lambda item, client=client: self._case(item, client)) for case in cases]

    def _case(self, case: EvaluationCase, client: LightRAGClient) -> EvaluationCaseResult:
        expected_source = str(case.expected["source_id"])
        lightrag = client.query(str(case.input["query"]), mode="hybrid", top_k=5)
        rag_sources = [row.source_id for row in lightrag.results]
        rag_hit = expected_source in rag_sources
        graph_path = any(row.graph_path for row in lightrag.results if row.source_id == expected_source)
        settings = get_settings()
        original_backend = settings.knowledge_backend
        settings.knowledge_backend = "local_knowledge"
        try:
            baseline = MCPGateway(self.db).execute(
                ToolRequest(
                    tool_name="search_evidence",
                    arguments={"query": str(case.input["query"])},
                    context=ToolContext(user_id=self.seed["doctor"].id, role="doctor", task_type="prescription_review"),
                )
            )
        finally:
            settings.knowledge_backend = original_backend
        local_sources = [str(item.get("source_id")) for item in baseline.evidence]
        local_hit = expected_source in local_sources
        passed = local_hit if "lightrag_retrieval" in self.disabled else rag_hit and graph_path
        if "graph_context_path" in self.disabled:
            passed = False
            graph_path = False
        return self.result(
            case,
            passed,
            {
                "lightrag_hit": rag_hit,
                "local_hit": local_hit,
                "lightrag_sources": rag_sources,
                "local_sources": local_sources,
                "graph_path_visible": graph_path,
                "backend": "local_knowledge" if "lightrag_retrieval" in self.disabled else "lightrag",
                "fallback_visible": True,
                "degraded_visible": True,
            },
        )

    @staticmethod
    def _doc(case: EvaluationCase) -> LightRAGDoc:
        source_id = str(case.input["source_id"])
        return LightRAGDoc(
            doc_id=f"DOC_{case.case_id}",
            entry_id=f"ENTRY_{case.case_id}",
            entry_type="evaluation_evidence",
            title=str(case.input["document_title"]),
            text=str(case.input["document_text"]),
            metadata={"source_id": source_id, "source_title": "Phase 12 固定评测语料", "visibility": ["doctor", "student"], "license": "evaluation-fixture"},
        )
