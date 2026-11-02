from app.core.config import get_settings
from app.db.models import KnowledgeEntry, LearningCase
from app.knowledge.qa_knowledge_index import build_qa_knowledge_index, search_qa_knowledge
from app.mcp.adapters import local_knowledge
from app.mcp.contracts import ToolContext, ToolRequest


def test_qa_index_retrieves_mtcmb_and_shennong_only_for_student(db_session, tmp_path):
    settings = get_settings()
    settings.qa_knowledge_enabled = True
    db_session.add(
        KnowledgeEntry(
            entry_id="QA_SHENNONG_TEST",
            source_id="SRC_SHENNONG_TCM_DATASET",
            entry_type="instruction_qa",
            title="黄芪有哪些功效",
            content="问题：黄芪有哪些功效？\n回答：可用于学习补气、固表等中医功效概念。",
            keywords_json=["黄芪"],
            source_title="ShenNong",
            source_type="fixture",
            metadata_json={
                "visibility": ["student"],
                "doctor_evidence_allowed": False,
                "student_visible": True,
                "source_record_id": "line_test",
            },
        )
    )
    db_session.add(
        LearningCase(
            title="MTCMB 教学问题",
            anonymized_case={
                "case_id": "CASE_MTCMB_TEST",
                "source_id": "SRC_MTCMB",
                "source_record_id": "test:1",
                "source_file": "data/test.jsonl",
                "deidentified": True,
                "review_status": "approved_training_only",
                "training_payload": {
                    "question": "善除有汗骨蒸的药物是什么？",
                    "options": {"A": "知母", "B": "青蒿", "C": "地骨皮"},
                    "answer": "C",
                    "original_source": "测试题库",
                },
            },
            standard_analysis="C",
            risk_points=[],
            evidence_chain=[],
        )
    )
    db_session.flush()
    path = tmp_path / "qa.db"

    report = build_qa_knowledge_index(db_session, index_path=path)
    shennong = search_qa_knowledge(query="黄芪有哪些功效", role="student", limit=3, index_path=path)
    mtcmb = search_qa_knowledge(query="有汗骨蒸用什么药", role="student", limit=3, index_path=path)

    assert report["documents"] == 2
    assert shennong[0]["source_id"] == "SRC_SHENNONG_TCM_DATASET"
    assert mtcmb[0]["source_id"] == "SRC_MTCMB"
    assert "地骨皮" in mtcmb[0]["snippet"]
    assert search_qa_knowledge(query="黄芪有哪些功效", role="doctor", limit=3, index_path=path) == []
    assert search_qa_knowledge(query="黄芪有哪些功效", role="patient", limit=3, index_path=path) == []
    assert search_qa_knowledge(query="当我给患者看病的时候，步骤是什么", role="student", limit=3, index_path=path) == []


def test_local_knowledge_augments_student_search_with_qa_index(db_session, tmp_path):
    settings = get_settings()
    original_path = settings.qa_knowledge_index_path
    original_rerank = settings.qa_knowledge_rerank_enabled
    try:
        settings.qa_knowledge_enabled = True
        settings.qa_knowledge_index_path = str(tmp_path / "qa.db")
        settings.qa_knowledge_rerank_enabled = False
        db_session.add(
            KnowledgeEntry(
                entry_id="QA_STUDENT_SEARCH",
                source_id="SRC_SHENNONG_TCM_DATASET",
                entry_type="instruction_qa",
                title="四气五味如何理解",
                content="问题：四气五味如何理解？\n回答：这是中药药性理论中的基础概念。",
                keywords_json=["四气五味"],
                source_title="ShenNong",
                source_type="fixture",
                metadata_json={"visibility": ["student"], "doctor_evidence_allowed": False, "student_visible": True},
            )
        )
        db_session.flush()
        build_qa_knowledge_index(db_session)
        request = ToolRequest(
            tool_name="search_evidence",
            arguments={"query": "四气五味如何理解"},
            context=ToolContext(user_id=1, role="student", task_type="student_ai_chat"),
        )

        result = local_knowledge.execute(db_session, request)

        assert result["evidence"][0]["entry_id"] == "QA_STUDENT_SEARCH"
        assert result["__metadata__"]["qa_retrieval_count"] == 1
    finally:
        settings.qa_knowledge_index_path = original_path
        settings.qa_knowledge_rerank_enabled = original_rerank
