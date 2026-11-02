from app.db.models import LightRAGDocManifest
from app.lightrag_integration.provenance import ProvenanceResolver


def test_graph_provenance_uses_database_manifest_when_input_docs_are_missing(db_session, tmp_path):
    doc_id = "DOC_HERB_TEST_HUANGQI"
    db_session.add(
        LightRAGDocManifest(
            doc_id=doc_id,
            snapshot_id="SNAP_TEST",
            file_path="herbs.json",
            entry_ids_json=["HERB_TEST_HUANGQI"],
            source_ids_json=["SRC_TCM_DATA_FOR_ML"],
            visibility_json=["doctor", "student"],
            license_json={"SRC_TCM_DATA_FOR_ML": "MIT"},
            content_hash="test-content-hash",
        )
    )
    db_session.commit()

    result = ProvenanceResolver(
        docs_path=tmp_path / "missing.jsonl",
        db=db_session,
    ).resolve_graph(
        {
            "nodes": [
                {
                    "id": "黄芪",
                    "labels": ["黄芪"],
                    "properties": {
                        "entity_id": "黄芪",
                        "entity_type": "artifact",
                        "file_path": doc_id,
                    },
                }
            ],
            "edges": [],
        },
        role="doctor",
        max_depth=2,
        max_nodes=120,
    )

    assert len(result.nodes) == 1
    assert result.nodes[0].source_ids == ["SRC_TCM_DATA_FOR_ML"]
    assert result.nodes[0].visibility == ["doctor", "student"]

