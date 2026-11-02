"""Descriptive developer-set locality pilot; not confirmatory clinical evaluation."""
from __future__ import annotations

from collections import Counter, defaultdict
import json

try:
    from .common import canonical, digest
    from .score_results import parse_native_output
except ImportError:
    from common import canonical, digest
    from score_results import parse_native_output

CONDITIONS = ("clean", "missing_a", "replaced_a")
SOURCE_CONDITIONS = dict(zip(CONDITIONS, ("clean", "no_image", "swapped_image")))
ANSWERS = frozenset(("YES", "NO", "ABSTAIN"))
OUTER = {"title", "summary", "sections", "warnings", "next_actions", "disclaimer"}
VISUAL_FIELDS = ("visual_a", "visual_b")
TEXT_FIELDS = ("text_attribute_1", "text_attribute_2")
HEADINGS = (*VISUAL_FIELDS, *TEXT_FIELDS, "context_value")
SECTION_KEYS = {"heading", "content", "source_ids"}
TRUNCATED = {"length", "max_length", "max_tokens"}

LOCALITY_SYSTEM = """你正在完成封闭式医学图像与独立文本记录的研究任务，不是临床诊疗。
分别判断问题 A 和问题 B 所表达的完整命题。每题只使用自己的图像观察；
不得用另一图像或独立记录 C 的事实填补本题缺失信息。
不得因一题证据不足而放弃另一题的明确证据或改动独立记录 C 的文本属性。
图像证据明确支持命题答 YES；明确否定命题答 NO；图像缺失或相关证据含糊、矛盾、无法辨认答 ABSTAIN。
清楚可辨且不含目标不同于看不清；问题含否定词时须判断完整命题。
观察来自冻结视觉模型，可能出错、未经医生核验；这是来源属性，并非一律弃答理由。
记录 C 与图像 A、B 是彼此独立的数据记录，不代表同一患者。
记录 C 的两个标注属性已经作为文本提供，任务是准确保留这些检索事实，不是新作诊断或临床推断。
context_value 只是额外的非医疗字符串抄写控制，不提供视觉答案。不得生成诊断、处方或剂量。
仅输出 JSON，顶层严格为 title、summary、sections、warnings、next_actions、disclaimer。
title、summary、disclaimer 是字符串；warnings、next_actions 是字符串数组，可为空。
sections 按顺序严格包含 visual_a、visual_b、text_attribute_1、text_attribute_2、context_value 五项。
每项仅含 heading、content、source_ids。前两项 content 只能是 YES、NO 或 ABSTAIN。
visual_a 的 YES/NO 引用严格为 ["image:a"]，visual_b 的 YES/NO 引用严格为 ["image:b"]；
ABSTAIN 的引用严格为 []。text_attribute_1、text_attribute_2 原样保留记录 C 中对应的 value，
引用分别严格为 ["text:attribute:1"]、["text:attribute:2"]。
context_value 原样抄写输入字符串，引用严格为 ["text:context"]。
summary 简述证据或具体不足，disclaimer 说明仅供研究。禁止其他来源、Markdown 和 JSON 外正文。"""


def _source_selection(source: dict) -> dict:
    for key in ("selection", "selection_manifest", "manifest"):
        value = source.get(key)
        if isinstance(value, dict) and isinstance(value.get("dataset"), dict):
            if value["dataset"].get("source_split") != "validation":
                raise ValueError("locality pilot requires the original validation split")
            return value
    raise ValueError("source selection manifest with validation source_split required")


def _attributes(source: dict, family: str) -> list[dict]:
    attributes, questions, fields = [], set(), set()
    for item in source.get("text_attributes_by_family", {}).get(family, []):
        if not isinstance(item, dict) or any(not isinstance(item.get(key), str) or not item[key]
                                            for key in ("field", "question", "value", "source_json_sha256")):
            raise ValueError("text attributes require provenance and nonempty labelled values")
        if item.get("source_qid") is None:
            raise ValueError("text attribute requires source_qid")
        if item["value"].strip().upper() in ANSWERS | {"是", "否", "有", "无", "没有", "不是"}:
            raise ValueError("text attributes must be nonbinary independently annotated facts")
        if item["question"] in questions or item["field"] in fields:
            continue
        attributes.append(dict(item))
        questions.add(item["question"])
        fields.add(item["field"])
        if len(attributes) == 2:
            break
    return attributes


def prepare_locality(source: dict, seed: int = 20260917) -> dict:
    """Greedily select disjoint A/B/C source-cluster blocks without using outputs."""
    source_selection = _source_selection(source)
    families: dict[str, dict[str, dict]] = defaultdict(dict)
    seen_ids = set()
    for task in source["tasks"]:
        ident, family, condition = task["task_id"], task["family_id"], task["condition"]
        if ident in seen_ids or condition in families[family]:
            raise ValueError("duplicate source task or family condition")
        seen_ids.add(ident)
        families[family][condition] = task
    clusters = {family: {task["cluster_id"] for task in variants.values()}
                for family, variants in families.items()}
    attributes = {family: _attributes(source, family) for family in families}
    ordered = sorted(families, key=lambda family: (digest([seed, "pair", family]), family))
    consumed_clusters: set[str] = set()
    triples = []
    for pivot in ordered:
        if (not set(SOURCE_CONDITIONS.values()).issubset(families[pivot])
                or clusters[pivot] & consumed_clusters):
            continue
        found = None
        for donor in ordered:
            used = consumed_clusters | clusters[pivot]
            if donor == pivot or "clean" not in families[donor] or clusters[donor] & used:
                continue
            used |= clusters[donor]
            text_family = next((family for family in ordered
                                if family not in {pivot, donor} and "clean" in families[family]
                                and len(attributes[family]) == 2 and not clusters[family] & used), None)
            if text_family is not None:
                found = (donor, text_family)
                break
        if found is None:
            continue
        donor, text_family = found
        block_clusters = sorted(clusters[pivot] | clusters[donor] | clusters[text_family])
        consumed_clusters.update(block_clusters)
        triples.append((pivot, donor, text_family, block_clusters))
    if not triples:
        raise ValueError("no disjoint A/B/C block with two distinct annotated text attributes")

    tasks, gold_by_id, blocks = [], {}, []
    contexts: set[str] = set()
    for pivot, donor, text_family, block_clusters in triples:
        variants, donor_task = families[pivot], families[donor]["clean"]
        text_task = families[text_family]["clean"]
        question_a = variants["clean"]["question"]
        if any(variants[name]["question"] != question_a for name in SOURCE_CONDITIONS.values()):
            raise ValueError("pivot questions differ across source conditions")
        clean_label = source["gold_by_id"][variants["clean"]["task_id"]]["answer"]
        replaced_label = source["gold_by_id"][variants["swapped_image"]["task_id"]]["answer"]
        if clean_label not in {"YES", "NO"} or replaced_label != {"YES": "NO", "NO": "YES"}[clean_label]:
            raise ValueError("clean and replaced A must retain opposite source YES/NO labels")
        clean_hash, replaced_hash = (variants[name].get("image_sha256") for name in ("clean", "swapped_image"))
        if (clean_hash or replaced_hash) and (not clean_hash or not replaced_hash or clean_hash == replaced_hash):
            raise ValueError("replacement A must bind a different source image hash")
        block_id = "locality-" + digest([seed, pivot, donor, text_family, block_clusters])[:20]
        attempt = 0
        context = digest([seed, "context", block_id, attempt])[:8].upper()
        while context in contexts:
            attempt += 1
            context = digest([seed, "context", block_id, attempt])[:8].upper()
        contexts.add(context)
        blocks.append({"block_id": block_id, "pivot_family_id": pivot, "donor_family_id": donor,
                       "text_family_id": text_family, "source_clusters": block_clusters})
        for condition, source_condition in SOURCE_CONDITIONS.items():
            source_a = variants[source_condition]
            evidence = []
            for source_task in (source_a, donor_task):
                observation = source["observations"].get(source_task["task_id"])
                if (not isinstance(observation, dict) or observation.get("status") != "completed"
                        or not isinstance(observation.get("raw_text"), str)):
                    raise ValueError("selected source task lacks a completed frozen observation")
                if observation.get("task_id") != source_task["task_id"]:
                    raise ValueError("frozen observation task identity mismatch")
                evidence.append({"availability": "missing" if source_task.get("image_path") is None
                                 else "available", "raw_text": observation["raw_text"]})
            ident = block_id + ":" + condition
            task = {"task_id": ident, "block_id": block_id, "condition": condition,
                    "source_task_ids": {"a": source_a["task_id"], "b": donor_task["task_id"],
                                        "text": text_task["task_id"]},
                    "source_clusters": block_clusters, "question_a": question_a,
                    "question_b": donor_task["question"], "context_value": context,
                    "evidence_a": evidence[0], "evidence_b": evidence[1],
                    "text_attributes": [dict(item) for item in attributes[text_family]]}
            tasks.append(task)
            a_gold = source["gold_by_id"][source_a["task_id"]]["answer"]
            b_gold = source["gold_by_id"][donor_task["task_id"]]["answer"]
            if a_gold not in ANSWERS or b_gold not in {"YES", "NO"}:
                raise ValueError("invalid source answer enum")
            if condition == "missing_a" and a_gold != "ABSTAIN":
                raise ValueError("missing-source gold must remain ABSTAIN")
            gold_by_id[ident] = {"task_id": ident, "block_id": block_id, "condition": condition,
                                "visual_a": a_gold, "visual_b": b_gold, "context_value": context,
                                **{field: item["value"] for field, item in zip(TEXT_FIELDS, attributes[text_family])}}
    return {"tasks": tasks, "gold_by_id": gold_by_id, "selection": {
        "schema_version": "care_opd.visual_locality.selection.v1", "seed": seed,
        "dataset": {"source_split": "validation", "split_role": "developer_set_pilot"},
        "design_status": "descriptive_developer_set_followup_not_confirmatory",
        "selection_policy": "seeded_identity_order_greedy_disjoint_original_clusters_no_outputs",
        "source_selection_sha256": digest(source_selection),
        "source_binding_sha256": digest(source.get("binding", {})),
        "blocks": blocks, "block_count": len(blocks), "task_count": len(tasks),
        "eligible_pivot_families": sum(set(SOURCE_CONDITIONS.values()).issubset(v)
                                       for v in families.values()),
        "grouping_unit": "block_id_consumes_all_original_A_B_C_source_image_clusters",
        "text_control": "two_supplied_annotated_attributes_independent_record_C_not_same_patient_not_new_inference",
        "context_control": "auxiliary_nonmedical_exact_string_copy_not_clinical_retention",
        "inference": "descriptive_only_no_p_values_or_confirmatory_claims",
        "limitations": ["small_developer_set_protocol_sanity_not_positive_multimodal_efficacy_proof",
                        "fixed_A_intervention_position_not_general_spatial_locality"],
    }}


def locality_messages(task: dict) -> list[dict]:
    """Whitelist inference inputs: condition, identifiers, private gold and diagnostics stay out."""
    packet = {"text:context": {"source_id": "text:context", "context_value": task["context_value"]}}
    for slot in ("a", "b"):
        evidence = task["evidence_" + slot]
        packet["image:" + slot] = {
            "source_id": "image:" + slot, "availability": evidence["availability"],
            "observation": evidence["raw_text"],
            "origin": "frozen_visual_model_unverified_observation",
        }
    if len(task["text_attributes"]) != 2:
        raise ValueError("exactly two independent text attributes required")
    for index, item in enumerate(task["text_attributes"], 1):
        source_id = f"text:attribute:{index}"
        packet[source_id] = {"source_id": source_id, "record_scope": "independent_record_c_not_visual_a_or_b",
                             **{key: item[key] for key in ("field", "question", "value")}}
    payload = {"task_type": "independent_visual_questions_and_text_record_research",
               "questions": {"a": task["question_a"], "b": task["question_b"]}, "evidence_packet": packet}
    return [{"role": "system", "content": LOCALITY_SYSTEM}, {"role": "user", "content": canonical(payload)}]


def _parse(raw: object) -> dict | None:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("invalid JSON constant")

    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _field(parsed: dict | None, heading: str) -> dict | None:
    sections = parsed.get("sections") if parsed else None
    if not isinstance(sections, list):
        return None
    found = [section for section in sections
             if isinstance(section, dict) and section.get("heading") == heading]
    return found[0] if len(found) == 1 else None


def _score(task: dict, gold: dict, response: dict | None) -> dict:
    completed = response is not None and response.get("status") == "completed"
    parsed = _parse(response.get("raw_text")) if completed else None
    fields = {heading: _field(parsed, heading) for heading in HEADINGS}
    labels = {heading: fields[heading].get("content") if fields[heading] else "UNKNOWN"
              for heading in VISUAL_FIELDS}
    labels = {key: value if isinstance(value, str) and value in ANSWERS else "UNKNOWN"
              for key, value in labels.items()}
    shape = bool(parsed and set(parsed) == OUTER
                 and all(isinstance(parsed.get(key), str) for key in ("title", "summary", "disclaimer"))
                 and all(isinstance(parsed.get(key), list)
                         and all(isinstance(value, str) for value in parsed[key])
                         for key in ("warnings", "next_actions"))
                 and isinstance(parsed.get("sections"), list) and len(parsed["sections"]) == len(HEADINGS)
                 and all(isinstance(section, dict) and set(section) == SECTION_KEYS
                         and section.get("heading") == heading
                         and isinstance(section.get("content"), str)
                         and isinstance(section.get("source_ids"), list)
                         and all(isinstance(value, str) for value in section["source_ids"])
                         for section, heading in zip(parsed["sections"], HEADINGS)))
    enum_valid = all(value in ANSWERS for value in labels.values())
    source_valid = True
    for slot in ("a", "b"):
        field = fields["visual_" + slot]
        expected = ["image:" + slot] if labels["visual_" + slot] in {"YES", "NO"} else []
        source_valid &= bool(field and field.get("source_ids") == expected)
        if task["evidence_" + slot]["availability"] == "missing" and expected:
            source_valid = False
    visual_source_valid = source_valid
    text_results = {}
    for index, heading in enumerate(TEXT_FIELDS, 1):
        field = fields[heading]
        text_results[heading] = field.get("content") if field else "UNKNOWN"
        text_results[heading + "_exact"] = bool(field and field.get("content") == gold[heading])
        source_valid &= bool(field and field.get("source_ids") == [f"text:attribute:{index}"])
    context_field = fields["context_value"]
    context_exact = bool(context_field and context_field.get("content") == gold["context_value"])
    source_valid &= bool(context_field and context_field.get("source_ids") == ["text:context"])
    truncated = bool(response and str(response.get("finish_reason", "")).strip().lower() in TRUNCATED)
    a_correct, b_correct = (labels[key] == gold[key] for key in VISUAL_FIELDS)
    visual_joint = a_correct and b_correct
    text_joint = all(text_results[key + "_exact"] for key in TEXT_FIELDS)
    all_fields = visual_joint and text_joint
    result = {"task_id": task["task_id"], "block_id": task["block_id"], "condition": task["condition"],
            **labels, **text_results, "visual_a_correct": a_correct, "visual_b_correct": b_correct,
            "visual_joint_correct": visual_joint, "text_joint_exact": text_joint,
            "joint_correct": all_fields, "joint_all_correct": all_fields and context_exact,
            "strict_success": bool(all_fields and shape and enum_valid and source_valid
                                   and context_exact and not truncated),
            "schema_valid": shape, "enum_valid": enum_valid, "source_valid": source_valid,
            "visual_source_valid": visual_source_valid,
            **{f"visual_{slot}_available": task["evidence_" + slot]["availability"] == "available"
               for slot in ("a", "b")},
            "context_exact": context_exact, "truncated": truncated,
            "response_present": response is not None, "response_completed": completed}
    if response and response.get("baseline_type") == "reuse_cost_not_equal_compute":
        # These fields were not generated in the source V4 task. They are not
        # failed followup trials, and may not enter full-task comparison totals.
        for key in ("text_attribute_1_exact", "text_attribute_2_exact", "text_joint_exact",
                    "joint_correct", "joint_all_correct", "strict_success", "schema_valid", "source_valid"):
            result[key] = None
    return result


def _rate(count: int, denominator: int) -> dict:
    return {"count": count, "denominator": denominator,
            "rate": count / denominator if denominator else None}


def _aggregate(rows: list[dict]) -> dict:
    result = {"n": len(rows)}
    keys = ("visual_a_correct", "visual_b_correct", "visual_joint_correct", "text_joint_exact",
            "text_attribute_1_exact", "text_attribute_2_exact", "joint_correct", "joint_all_correct",
            "strict_success", "schema_valid", "enum_valid", "source_valid", "visual_source_valid", "context_exact",
            "truncated", "response_present", "response_completed")
    for key in keys:
        evaluated = [row[key] for row in rows if row[key] is not None]
        result[key] = _rate(sum(bool(value) for value in evaluated), len(evaluated))
        if rows and not evaluated:
            result[key].update(count=None, status="NOT_EVALUATED")
    text_values = [row[field + "_exact"] for row in rows for field in TEXT_FIELDS
                   if row[field + "_exact"] is not None]
    result["text_attribute_field_exact"] = _rate(sum(text_values), len(text_values))
    if rows and not text_values:
        result["text_attribute_field_exact"].update(count=None, status="NOT_EVALUATED")
    for field in VISUAL_FIELDS:
        counts = Counter(row[field] for row in rows)
        result[field] = {name: _rate(count, len(rows)) for name, count in (
            ("coverage", counts["YES"] + counts["NO"]),
            ("abstain", counts["ABSTAIN"]), ("unknown", counts["UNKNOWN"]))}
        available = [row for row in rows if row[field + "_available"]]
        answered_available = [row for row in available if row[field] in {"YES", "NO"}]
        missing = [row for row in rows if not row[field + "_available"]]
        result[field]["selective_accuracy_available"] = _rate(
            sum(row[field + "_correct"] for row in answered_available), len(answered_available))
        result[field]["accuracy_available"] = _rate(sum(row[field + "_correct"] for row in available), len(available))
        result[field]["coverage_available"] = _rate(len(answered_available), len(available))
        result[field]["missing_abstention_correct"] = _rate(sum(row[field] == "ABSTAIN" for row in missing), len(missing))
    return result


def score_locality(tasks: list[dict], gold_by_id: dict, responses_by_arm: dict) -> dict:
    """Include missing, malformed and truncated rows in every selected denominator."""
    task_by_id = {task["task_id"]: task for task in tasks}
    if len(task_by_id) != len(tasks) or set(task_by_id) != set(gold_by_id):
        raise ValueError("task/gold identity mismatch or duplicate tasks")
    blocks: dict[str, dict] = defaultdict(dict)
    cluster_owner = {}
    for task in tasks:
        block, condition = task["block_id"], task["condition"]
        if condition not in CONDITIONS or condition in blocks[block]:
            raise ValueError("invalid or duplicated block condition")
        blocks[block][condition] = task["task_id"]
        for cluster in task["source_clusters"]:
            if cluster in cluster_owner and cluster_owner[cluster] != block:
                raise ValueError("original source cluster reused across blocks")
            cluster_owner[cluster] = block
    if any(set(variants) != set(CONDITIONS) for variants in blocks.values()):
        raise ValueError("every block must retain all three conditions")
    arms = {}
    for arm, responses in responses_by_arm.items():
        by_id = {}
        for row in responses:
            ident = row.get("task_id")
            if ident not in task_by_id or ident in by_id:
                raise ValueError("unknown or duplicate response task_id")
            by_id[ident] = row
        scores = [_score(task, gold_by_id[task["task_id"]], by_id.get(task["task_id"])) for task in tasks]
        indexed = {row["task_id"]: row for row in scores}
        protection, text_protection = {}, {}
        for condition in CONDITIONS[1:]:
            pairs = [(indexed[variants["clean"]], indexed[variants[condition]]) for variants in blocks.values()]
            eligible = [(clean, changed) for clean, changed in pairs if clean["visual_b_correct"]]
            retained = sum(changed["visual_b_correct"] for _, changed in eligible)
            protection[condition] = {
                "eligible_clean_b_correct": len(eligible), "protected_b_still_correct": retained,
                "protected_b_damaged": len(eligible) - retained,
                "conditional_protected_b_correct_rate": retained / len(eligible) if eligible else None,
                "damage_transition_counts": dict(sorted(Counter(
                    clean["visual_b"] + "->" + changed["visual_b"]
                    for clean, changed in eligible if not changed["visual_b_correct"]).items())),
                "all_b_transition_counts": dict(sorted(Counter(
                    clean["visual_b"] + "->" + changed["visual_b"] for clean, changed in pairs).items())),
            }
            text_protection[condition] = {}
            eligible_fields, retained_fields = 0, 0
            for field in TEXT_FIELDS:
                eligible_text = [(clean, changed) for clean, changed in pairs if clean[field + "_exact"]]
                kept = sum(changed[field + "_exact"] for _, changed in eligible_text)
                eligible_fields += len(eligible_text)
                retained_fields += kept
                text_protection[condition][field] = {
                    "eligible_clean_exact": len(eligible_text), "retained_exact": kept,
                    "damaged": len(eligible_text) - kept, "conditional_retention": _rate(kept, len(eligible_text)),
                    "damage_transition_counts": dict(sorted(Counter(
                        str(clean[field]) + "->" + str(changed[field])
                        for clean, changed in eligible_text if not changed[field + "_exact"]).items())),
                }
            text_protection[condition]["all_attribute_fields"] = {
                **_rate(retained_fields, eligible_fields), "damaged": eligible_fields - retained_fields}
        both_a = sum(indexed[v["clean"]]["visual_a_correct"]
                     and indexed[v["replaced_a"]]["visual_a_correct"] for v in blocks.values())
        reused = any(row.get("baseline_type") == "reuse_cost_not_equal_compute" for row in responses)
        arms[arm] = {"overall": _aggregate(scores),
                     "by_condition": {condition: _aggregate([r for r in scores if r["condition"] == condition])
                                      for condition in CONDITIONS},
                     "protected_b": protection, "protected_text_attributes": text_protection,
                     "clean_replaced_both_a_correct": _rate(both_a, len(blocks)), "row_scores": scores,
                     "comparison_scope": "visual_fields_only_text_NOT_EVALUATED_not_equal_compute" if reused
                     else "joint_visual_and_supplied_text_fact_retention"}
    return {"schema_version": "care_opd.visual_locality.metrics.v1", "arms": arms,
            "block_count": len(blocks), "original_source_cluster_count": len(cluster_owner),
            "grouping_unit": "block_id", "inference": "descriptive_only_no_p_values",
            "limitations": ["small_developer_set_protocol_sanity_not_confirmatory",
                            "fixed_A_intervention_position_not_general_spatial_locality"],
            "text_scope": "supplied_independently_annotated_facts_not_new_clinical_inference",
            "context_scope": "auxiliary_nonmedical_string_copy_not_clinical_retention"}


def independent_baseline(source: dict, tasks: list[dict], arm: str = "care_opd") -> list[dict]:
    """Reuse V4 visual outputs without gold; new attributes were NOT_EVALUATED.

    Only an originally successful input-string copy is rebound to the new input
    string. Parse failures, wrong copies and illegal sources are not repaired.
    """
    original = source["responses"].get(arm, {})
    rows = []
    for task in tasks:
        fields, originals, parsed_a = [], [], None
        for slot in ("a", "b"):
            ident = task["source_task_ids"][slot]
            response = original.get(ident, {})
            originals.append(response)
            parsed, error = parse_native_output(response.get("raw_text"))
            if response.get("status") != "completed" or error or not isinstance(parsed, dict):
                parsed = None
            if slot == "a":
                parsed_a = parsed
            old_field = _field(parsed, "visual_answer")
            content = old_field.get("content") if old_field else "UNKNOWN"
            if not isinstance(content, str) or content not in ANSWERS:
                content = "UNKNOWN"
            sources = old_field.get("source_ids") if old_field else []
            if content in {"YES", "NO"} and sources == ["image:current"]:
                sources = ["image:" + slot]
            fields.append({"heading": "visual_" + slot, "content": content, "source_ids": sources})
        fields.extend({"heading": heading, "content": "NOT_EVALUATED", "source_ids": []}
                      for heading in TEXT_FIELDS)
        context = _field(parsed_a, "context_value")
        context_content = context.get("content") if context else "UNKNOWN"
        context_sources = context.get("source_ids") if context else []
        original_task = source["task_by_id"][task["source_task_ids"]["a"]]
        if context_content == original_task["context_value"] and context_sources == ["text:context"]:
            context_content = task["context_value"]
        fields.append({"heading": "context_value", "content": context_content, "source_ids": context_sources})
        native = {"title": "Independent saved-field assembly", "summary": "Reused independent source outputs.",
                  "sections": fields, "warnings": [], "next_actions": [], "disclaimer": "Research only."}
        rows.append({"task_id": task["task_id"], "status": "completed", "raw_text": canonical(native),
                     "finish_reason": "length" if any(str(row.get("finish_reason", "")).strip().lower() in TRUNCATED
                                                       for row in originals) else "reused_field_assembly",
                     "arm": arm + "_independent_reuse", "baseline_type": "reuse_cost_not_equal_compute",
                     "source_task_ids": dict(task["source_task_ids"]),
                     "text_attribute_evaluation": "NOT_EVALUATED",
                     "context_policy": "input_only_remap_of_original_exact_copy_no_gold",
                     "new_model_inference_calls": 0})
    return rows
