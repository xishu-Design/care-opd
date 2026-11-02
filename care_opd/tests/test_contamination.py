import json

from care_opd.contamination import audit


def _dump(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_exact_overlap_is_rejected(tmp_path):
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    _dump(train, [{"prompt": "患者甲的独立合成病例", "target": "动作A"}])
    _dump(test, [{"question": "患者甲的独立合成病例", "analysis": "动作A"}])
    report = audit(
        train,
        test,
        train_fields=("prompt", "target"),
        eval_fields=("question", "analysis"),
        ngram=4,
    )
    assert not report["passed"]
    assert len(report["exact_matches"]) == 1

