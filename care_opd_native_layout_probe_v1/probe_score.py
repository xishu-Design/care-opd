"""Gold-aware CPU-only comparison of cached and native-layout first outputs.

This module is imported only by the score command, never by inference. It uses
the frozen original scorer without normalizing observations or repairing output.
All numbers describe a post-hoc development diagnostic, not new test evidence.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path


VERSION = "care_opd.native_layout_probe.score.v1"
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260918
LAYOUTS = ("old_cached", "native_layout")


def _name(arm, layout):
    return arm + "__" + layout


def _flatten_rates(value, prefix=""):
    """Retain every original rate and its explicit denominator, without rows."""
    if not isinstance(value, dict):
        return
    if {"count", "denominator", "rate"} <= set(value):
        yield prefix, value
        return
    for key, item in value.items():
        if key != "rows":
            yield from _flatten_rates(item, prefix + "/" + key if prefix else key)


def _metric_tables(report, arms):
    metrics, comparisons = [], []
    for arm in arms:
        for view in ("strict", "semantic"):
            layouts = {
                layout: dict(_flatten_rates(report["arms"][_name(arm, layout)][view]))
                for layout in LAYOUTS
            }
            if set(layouts["old_cached"]) != set(layouts["native_layout"]):
                raise ValueError("original scorer metric paths differ across layouts")
            for path in sorted(layouts["old_cached"]):
                old, new = (layouts[layout][path] for layout in LAYOUTS)
                for layout, metric in zip(LAYOUTS, (old, new)):
                    metrics.append({"model": arm, "layout": layout, "view": view,
                                    "metric": path, **metric})
                comparisons.append({
                    "model": arm, "view": view, "metric": path,
                    "old_count": old["count"], "old_denominator": old["denominator"],
                    "old_rate": old["rate"], "new_count": new["count"],
                    "new_denominator": new["denominator"], "new_rate": new["rate"],
                    "new_minus_old": (new["rate"] - old["rate"]
                                      if old["rate"] is not None and new["rate"] is not None else None),
                    "paired_inference": "see paired_layout_effects.csv only for scorer-supported endpoints",
                })
    return metrics, comparisons


def _paired_layout_effects(report, arms):
    """Orient unchanged original paired bootstrap estimates as native minus old.

    Other cross-model/layout contrasts remain in the original metrics.json but
    are not presented as the layout probe's paired contrasts.
    """
    effects = []
    for arm in arms:
        old, new = _name(arm, "old_cached"), _name(arm, "native_layout")
        found = False
        for item in report["paired_differences"]:
            if {item["arm"], item["reference"]} != {old, new}:
                continue
            found = True
            new_first = item["arm"] == new
            left, right = ("arm", "reference") if new_first else ("reference", "arm")
            difference = item["difference"]
            ci = item["ci95"]
            if not new_first:
                difference = -difference if difference is not None else None
                ci = [-ci[1] if ci[1] is not None else None,
                      -ci[0] if ci[0] is not None else None]
            effects.append({
                "model": arm, "view": item["view"], "metric": item["metric"],
                "contrast": "native_layout_minus_old_cached",
                "new_count": item[left + "_count"],
                "new_denominator": item[left + "_denominator"],
                "old_count": item[right + "_count"],
                "old_denominator": item[right + "_denominator"],
                "difference": difference, "ci95_low": ci[0], "ci95_high": ci[1],
                "bootstrap_valid_samples": item["bootstrap_valid_samples"],
                "bootstrap_samples": item["bootstrap_samples"],
                "independent_blocks": item["independent_blocks"],
                "interpretation": "post_hoc_development_diagnostic_not_confirmatory",
            })
        if not found:
            raise ValueError("missing same-model original paired effects: " + arm)
    return effects


def _csv_bytes(rows):
    if not rows:
        raise ValueError("refusing empty diagnostic table")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "NA" if value is None else
                         json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                         if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    return stream.getvalue().encode("utf-8-sig")


def _show(metric):
    count, total, value = metric["count"], metric["denominator"], metric["rate"]
    return f"{count}/{total}（NA）" if value is None else f"{count}/{total}（{value:.1%}）"


def _runtime_diagnostics(responses, arms):
    result = {}
    for arm in arms:
        for layout in LAYOUTS:
            key = _name(arm, layout)
            rows = list(responses[key].values())
            reasons = {}
            for row in rows:
                reason = row.get("finish_reason")
                reasons[reason] = reasons.get(reason, 0) + 1
            entry = {"rows": len(rows), "finish_reasons": reasons}
            for field in ("prompt_tokens", "completion_tokens", "latency_seconds"):
                observed = [row[field] for row in rows
                            if isinstance(row.get(field), (int, float)) and not isinstance(row[field], bool)]
                entry[field] = {"available_count": len(observed), "total_count": len(rows),
                                "min": min(observed) if observed else None,
                                "max": max(observed) if observed else None,
                                "mean": sum(observed) / len(observed) if observed else None}
            result[key] = entry
    return {"schema_version": VERSION, "arms": result,
            "latency_comparison_warning": "Cached old and newly generated native outputs may differ in hardware/batch/time; not a controlled speed benchmark."}


def _summary(report, arms, scoring_manifest):
    lines = ["# 原生证据布局探针：旧缓存与新输入对照", "",
             "这是事后开发集诊断，不是新的独立验证，也不以结果自动判定方法通过或失败。", "",
             f"纳入 {report['block_count']} 个完整区组、每组6条件，共 {report['task_count']} 条任务；"
             "四个模型均保留同一任务集的旧缓存和新布局首个输出。仅将证据改放到"
             "evidence_packet.retrieved_evidence，系统指令、问题、record_context与输出合同保持不变。"
             "复用视觉观察，不新增VLM推理。", "",
             "## 正常证据下的有效回答", "",
             "正常条件为 clean、benign_overlay、valid_update；每条任务的A/B分别进入分母。"
             "正确率、覆盖率、已回答部分正确率同时展示，弃答不被当作有效回答。", ""]
    for view in ("strict", "semantic"):
        lines += [f"### {view} 视图", "",
                  "|模型|输入布局|正常答案正确率|回答覆盖率|已回答部分正确率|全条件联合有效率|JSON/模式有效|截断|",
                  "|---|---|---|---|---|---|---|---|"]
        for arm in arms:
            for layout in LAYOUTS:
                overall = report["arms"][_name(arm, layout)][view]["overall"]
                cells = [arm, layout] + [_show(overall[key]) for key in
                    ("normal_accuracy", "normal_coverage", "normal_selective_accuracy",
                     "joint_visual_safe_useful")]
                cells += [_show(overall["json_valid"]) + " / " + _show(overall["schema_valid"]),
                          _show(overall["truncated"])]
                lines.append("|" + "|".join(cells) + "|")
        lines += [""]
    lines += ["## 有效换图与局部内容保持", "",
              "以下为 semantic 视图；strict及所有分条件指标完整保存在metrics.json和metrics.csv。"
              "新旧图都答对才计为有效更新。条件保持率的分母是该布局下clean B答对的区组数，"
              "新旧布局的分母可能不同，不能把不同条件子集视为同一批固定合格样本。", "",
              "|模型|输入布局|A新旧均正确|clean B正确|断源时B正确（无条件）|断源B保持（条件）|错配B保持（条件）|",
              "|---|---|---|---|---|---|---|"]
    for arm in arms:
        for layout in LAYOUTS:
            value = report["arms"][_name(arm, layout)]["semantic"]
            metrics = [value["pairs"]["valid_update"]["a_both_correct"],
                       value["by_condition"]["clean"]["b_accuracy"],
                       value["pairs"]["b_retention"]["source_outage"]["unconditional_correct"],
                       value["pairs"]["b_retention"]["source_outage"]["conditional_on_clean_correct"],
                       value["pairs"]["b_retention"]["scope_mismatch"]["conditional_on_clean_correct"]]
            lines.append("|" + "|".join([arm, layout] + [_show(m) for m in metrics]) + "|")
    lines += ["", "## 解释边界", "",
              "- 旧结果来自缓存，新结果在另一时间/运行批次生成，不能将观察差异直接等同于输入布局的纯因果效应。",
              "- 没有改答案标签、评分器、输出内容或分母；空输出、非法JSON、截断及弃答保留。",
              "- semantic仅允许完整Markdown围栏拆封，不能补字段、改答案、转换UNKNOWN或挑选更好输出。",
              "- 该探针只改变输入布局，仍不是原生训练任务的完整复现，也不能验证token责任区粒度已获得改善。",
              "- 全部区组先由无答案标签的固定规则选定，后评分；不能据本结果继续筛选更有利的子集作为独立测试。",
              "- 按区组配对bootstrap，共2000次，seed=20260918。小区组区间可能退化；0分母为NA，不是0%或100%。",
              "- 所有跨组对比保留在原评分器metrics.json；专用paired_layout_effects.csv只展示同模型新布局减旧布局。",
              "- 该结果不能单独证明多模态贡献成立、临床有效或满足期刊scope。", "",
              "## 文件", "",
              "- metrics.json：原评分器未删减的八组结果、配对区间及视觉前端诊断。",
              "- metrics.csv / layout_comparison.csv：全部原指标及显式分母，后者含新减旧描述差。",
              "- paired_layout_effects.csv：原评分器支持端点的同模型布局配对差。",
              "- row_diagnostics.private.csv：每任务原评分字段，用于逐条定位，不作为公开材料。",
              "- runtime_diagnostics.json：结束原因、输入输出长度及计时可用性。",
              "- scoring_manifest.json：输入、协议、评分源代码与输出身份绑定。", "",
              "评分身份：`" + scoring_manifest["scoring_sha256"] + "`。", ""]
    return "\n".join(lines)


def score(work):
    """Load checked frozen input, then and only then open source gold to score."""
    import probe_core as p
    import boundary_score as original

    work = Path(work).resolve()
    manifest, tasks, observations, prompts = p.load_work(work)
    task_list = list(tasks.values()) if isinstance(tasks, dict) else list(tasks)
    task_ids = {task["task_id"] for task in task_list}
    if len(task_ids) != len(task_list) or len(task_ids) != manifest["task_count"]:
        raise ValueError("duplicate tasks or manifest task_count mismatch")
    arms = tuple(p.c.ARMS)
    responses = {}
    for arm in arms:
        baseline_path = work / "baseline" / (arm + ".jsonl")
        baseline = p.c.index_rows(p.c.read_jsonl(baseline_path))
        current = p.load_new_rows(work, arm, require_complete=True)
        if set(baseline) != task_ids or set(current) != task_ids:
            raise ValueError("complete identical task sets required; no partial scoring: " + arm)
        responses[_name(arm, "old_cached")] = baseline
        responses[_name(arm, "native_layout")] = current

    gold_path = Path(manifest["source_root"]) / "gold.private.jsonl"
    if p.c.hash_file(gold_path) != manifest["gold_sha256"]:
        raise ValueError("source gold hash differs from frozen manifest")
    all_gold = p.c.index_rows(p.c.read_jsonl(gold_path))
    if not task_ids <= set(all_gold):
        raise ValueError("source gold missing selected tasks")
    gold = {ident: all_gold[ident] for ident in task_ids}
    if p.c.hash_file(gold_path) != manifest["gold_sha256"]:
        raise ValueError("source gold changed during read")
    task_observations = {task["task_id"]: p.c.task_observations(task, observations) for task in task_list}
    report = original.score_rows(task_list, gold, responses, task_observations,
                                 bootstrap_samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED)
    identity = {
        "schema_version": VERSION, "protocol_sha256": manifest["protocol_sha256"],
        "package_sha256": manifest["package_sha256"], "gold_sha256": manifest["gold_sha256"],
        "selected_blocks": manifest["selected_blocks"], "task_count": len(task_ids),
        "tasks_sha256": p.c.digest(sorted(task_list, key=lambda row: row["task_id"])),
        "observations_sha256": p.c.digest(task_observations),
        "prompts_sha256": p.c.digest(prompts),
        "responses_sha256": {name: p.c.digest(sorted(rows.values(), key=lambda row: row["task_id"]))
                             for name, rows in sorted(responses.items())},
        "original_scorer_sha256": p.c.hash_file(original.__file__),
        "probe_scorer_sha256": p.c.hash_file(__file__),
        "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
    }
    digest = p.c.digest(identity)
    scoring_manifest = {**identity, "scoring_sha256": digest,
                        "evidence_status": "POST_HOC_DEVELOPMENT_DIAGNOSTIC_NOT_INDEPENDENT_VALIDATION",
                        "new_vlm_inference": False, "output_selection": "first_completed_output_frozen",
                        "automatic_performance_pass_fail": False,
                        "old_new_run_time_batch_confound": True, "raw_outputs_repaired": False}
    output = work / "summary" / digest[:16]
    metrics, comparisons = _metric_tables(report, arms)
    effects = _paired_layout_effects(report, arms)
    diagnostic_rows = [dict(row, model=arm, layout=layout)
                       for arm in arms for layout in LAYOUTS for view in ("strict", "semantic")
                       for row in report["arms"][_name(arm, layout)][view]["rows"]]
    p.c.json_once(output / "metrics.json", report)
    for name, rows in (("metrics.csv", metrics), ("layout_comparison.csv", comparisons),
                       ("paired_layout_effects.csv", effects),
                       ("row_diagnostics.private.csv", diagnostic_rows)):
        p.c.write_once(output / name, _csv_bytes(rows))
    p.c.json_once(output / "runtime_diagnostics.json", _runtime_diagnostics(responses, arms))
    p.c.json_once(output / "scoring_manifest.json", scoring_manifest)
    p.c.write_once(output / "summary.md", _summary(report, arms, scoring_manifest).encode("utf-8"))
    return {"status": "SCORING_COMPLETE_NOT_A_PERFORMANCE_PASS", "summary": str(output),
            "task_count": len(task_ids), "block_count": report["block_count"],
            "model_count": len(arms), "layout_count": len(LAYOUTS), "scoring_sha256": digest}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    print(json.dumps(score(parser.parse_args().work), ensure_ascii=False, indent=2))
