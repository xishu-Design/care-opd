"""Derive protected text records from separate, source-annotated SLAKE QA.

These are explicitly supplied textual facts, NOT independently inferred image
answers. Locality pairing must use a third image cluster, distinct from A/B.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import unicodedata

import common
import prepare_slake


def normalized(value) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def attach_attributes(source: dict) -> dict:
    dataset = source["selection"]["dataset"]
    if dataset.get("source_split") != "validation":
        raise ValueError("protected attributes require the validation source")
    path = Path(dataset["source_json"])
    expected = dataset["source_json_sha256"]
    if common.file_sha256(path) != expected:
        raise ValueError("annotation source hash differs from the frozen selection")
    rows = common.read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("annotation source must be a nonempty JSON array")
    by_qid, by_image = {}, defaultdict(list)
    for row in rows:
        qid = normalized(row.get("qid", ""))
        if not qid or qid in by_qid:
            raise ValueError("missing or duplicate source annotation qid")
        by_qid[qid] = row
        by_image[prepare_slake._image_key(prepare_slake._image_reference(row))].append(row)
    result, diagnostics = {}, []
    for task in source["tasks"]:
        if task["condition"] != "clean":
            continue
        original = by_qid.get(normalized(task["dataset_qid"]))
        if original is None or normalized(original["question"]) != normalized(task["question"]):
            raise ValueError("prepared task is not bound to its original annotation")
        image_key = prepare_slake._image_key(prepare_slake._image_reference(original))
        candidates = []
        for row in by_image[image_key]:
            if str(row.get("base_type", "")).lower() != "vqa" or row.get("q_lang") != "zh":
                continue
            question, value = row.get("question"), row.get("answer")
            if not isinstance(question, str) or not isinstance(value, str) or not normalized(value):
                continue
            if prepare_slake.normalize_binary_answer(value) is not None:
                continue
            kind = normalized(row.get("content_type", ""))
            # A field is an independently labelled question, not an invented
            # clinical attribute. Missing category remains explicit in provenance.
            field = "slake:" + (kind or "question_" + common.digest(normalized(question))[:12])
            candidates.append({"field": field, "question": normalized(question),
                               "value": normalized(value), "source_qid": normalized(row["qid"]),
                               "source_json_sha256": expected, "source_image_key": image_key,
                               "annotation_category": kind or None})
        chosen, seen_fields, seen_questions = [], set(), set()
        for item in sorted(candidates, key=lambda x: (x["field"], x["source_qid"])):
            if item["field"] in seen_fields or item["question"] in seen_questions:
                continue
            # Conflicting labels for the same source question are not selected.
            values = {x["value"] for x in candidates if x["question"] == item["question"]}
            if len(values) != 1:
                continue
            chosen.append(item)
            seen_fields.add(item["field"])
            seen_questions.add(item["question"])
        result[task["family_id"]] = chosen
        diagnostics.append({"family_id": task["family_id"], "cluster_id": task["cluster_id"],
                            "eligible_distinct_fields": len(chosen)})
    if common.file_sha256(path) != expected:
        raise ValueError("annotation source changed during attribute preparation")
    source["text_attributes_by_family"] = result
    source["text_attribute_provenance"] = {
        "source_json": str(path.resolve()), "source_json_sha256": expected,
        "source_verification_status": dataset.get("verification_status", "unknown"),
        "selection_rule": "Chinese vqa nonbinary labels; distinct source category/question; lexical deterministic order",
        "input_semantics": "provided_text_record_C_not_visual_answer_gold_for_A_or_B",
        "not_independent_medical_adjudication": True, "family_inventory": diagnostics,
    }
    return source
