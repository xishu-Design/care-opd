from __future__ import annotations

from app.knowledge.snapshots import active_snapshot, create_snapshot, transition_snapshot
from scripts.build_local_ai_stack import _activate_artifacts


def test_failed_snapshot_does_not_replace_active_snapshot(db_session, tmp_path):
    content = tmp_path / "content"
    content.mkdir()
    (content / "herbs.json").write_text('{"items": []}', encoding="utf-8")
    first = create_snapshot(
        db_session,
        version="v1",
        source_manifest={},
        record_counts={},
        workspace="one",
        content_dir=content,
    )
    transition_snapshot(db_session, first.snapshot_id, "VERIFYING")
    transition_snapshot(db_session, first.snapshot_id, "ACTIVE")
    second = create_snapshot(
        db_session,
        version="v2",
        source_manifest={},
        record_counts={},
        workspace="two",
        content_dir=content,
    )
    transition_snapshot(db_session, second.snapshot_id, "FAILED", reason="pilot failed")

    current = active_snapshot(db_session)

    assert current is not None
    assert current.snapshot_id == first.snapshot_id
    assert second.failure_reason == "pilot failed"


def test_content_and_graph_activate_together(tmp_path):
    staged_content = tmp_path / "staged_content"
    staged_graph = tmp_path / "staged_graph"
    active_content = tmp_path / "active_content"
    active_graph = tmp_path / "active_graph"
    for path, value in (
        (staged_content, "new-content"),
        (staged_graph, "new-graph"),
        (active_content, "old-content"),
        (active_graph, "old-graph"),
    ):
        path.mkdir()
        (path / "value.txt").write_text(value, encoding="utf-8")

    _activate_artifacts(staged_content, active_content, staged_graph, active_graph)

    assert (active_content / "value.txt").read_text(encoding="utf-8") == "new-content"
    assert (active_graph / "value.txt").read_text(encoding="utf-8") == "new-graph"
