#!/usr/bin/env python3
"""Pure-CPU protocol and data loader for the TCMBench V2 evaluation.

The benchmark repository does not currently ship a licence file, so this
project deliberately stores no TCMBench question text.  Callers point this
module at an independently obtained official checkout.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATA_COMMIT = "7e2d2326ff79da186c6fccaafa374cf1f77ac968"
PIPELINE_REFERENCE_COMMIT = "f0138f30eff2ab8f5abcf07637fb3aeea8515c8a"
PROTOCOL_VERSION = "care-opd-tcmbench-v2-robustness-v1"

# Hashes are for the public data snapshot at DATA_COMMIT.  The evaluation can
# refuse silently changed labels/questions instead of producing incomparable
# numbers under the same benchmark name.
EXPECTED_FILES: dict[str, tuple[int, str]] = {
    "first_level/FKU.json": (1600, "10306988f78d0812a194cf3d2ebb251c67970d235f17b4ebaffd1859157fa0a1"),
    "first_level/KHC.json": (1481, "0765991b72381ae2b5462f18d47001d9297312faef84b5b53b8d99458ae6b819"),
    "first_level/CVR.json": (198, "a7e72dc9a833819d7de1a1a796a04ad4a4a90e9143e535663c33fc92c8ea6f1a"),
    "second_level/CBF.json": (137, "7365f7399b0865c654e610398ed9277c375c6fd2e806cb8bdd8cb1e78d8b2371"),
    "second_level/DDST.json": (211, "e7942291cbdd6eb367644fd12fc7311e8aa978bf67d0611ad4894805baceeda4"),
    "second_level/PF.json": (125, "7f8ff340f4ec5881ff40ab6565c0ae132ab3e5ae81714c9316c53499e59b0f93"),
    "second_level/SCF.json": (56, "41f9242aba5ec59c5860b29629436d91032bf687448f6809475c94cbaa3a989b"),
    "second_level/SDDT.json": (90, "00596b4f32496f5e3ef18b2ed6b83d261b493147a7825c4754fdeba5a37c46d5"),
    "second_level/SDT.json": (77, "ebf51489b382c57dea3bca2151b14ffecf5b4b8c6e678ef572bfb8345ec3270f"),
    "third_level/DDST_hard.json": (86, "3c1d6cc2f15270c59a18cb575a2320619d9f783288ef33b97798901844c116f5"),
    "third_level/SDDT_hard.json": (156, "df37871bf99214cb35e0c4dcec02d9d3e0263d9e9970f51a66d0995ea90f0c6c"),
    "third_level/SDT_reverse.json": (77, "eb4f4ff19cd29e4d8b3cdc2ddace1516caaba19ec81a071a9c244439019fe9c9"),
    "third_level/SDT_shuffle.json": (77, "e57c89ff2185917e6bc8196a6511ef5c9a2ee023867564b38ffa4d58a8469105"),
    "third_level/herb_predict.json": (239, "9a1da758eeeae1fe082170adca3981cf568e5cdb6333ea7048cc40cfa1783bd0"),
}

TASK_TO_FILE = {
    "FKU": "first_level/FKU.json",
    "KHC": "first_level/KHC.json",
    "CVR": "first_level/CVR.json",
    "CBF": "second_level/CBF.json",
    "DDST": "second_level/DDST.json",
    "PF": "second_level/PF.json",
    "SCF": "second_level/SCF.json",
    "SDDT": "second_level/SDDT.json",
    "SDT": "second_level/SDT.json",
    "DDST_hard": "third_level/DDST_hard.json",
    "SDDT_hard": "third_level/SDDT_hard.json",
    "SDT_reverse": "third_level/SDT_reverse.json",
    "SDT_shuffle": "third_level/SDT_shuffle.json",
    "herb_predict": "third_level/herb_predict.json",
}

ALL_13_TASKS = (
    "FKU", "KHC", "CVR", "CBF", "DDST", "PF", "SCF", "SDDT", "SDT",
    "DDST_hard", "SDDT_hard", "SDT_reverse", "SDT_shuffle",
)
ROBUSTNESS_TASKS = (
    "DDST", "SDDT", "SDT", "DDST_hard", "SDDT_hard",
    "SDT_reverse", "SDT_shuffle",
)
PRIMARY_VALID_TASKS = ("SDT", "SDT_reverse", "SDT_shuffle")
SEQUENTIAL_TASKS = {"KHC", "CVR", "SDT", "SDT_reverse", "SDT_shuffle"}
NUMERIC_TASKS = {"DDST_hard"}

_LEADING_SUBQUESTION = re.compile(r"^\s*\d+\s*[)）.．、:：]+\s*")
_OPTION_LINE = re.compile(r"(?m)^\s*([A-EＡ-Ｅ])\s*[.．、:：)]\s*")
_NUMBERED_LINE = re.compile(r"(?m)^\s*\[(\d+)\]")


def sha256_file(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split())


def normalize_subquestion(value: str) -> str:
    return normalize_text(_LEADING_SUBQUESTION.sub("", value))


def semantic_id(share_content: str, subquestion: str) -> str:
    return canonical_hash(
        {"share_content": normalize_text(share_content),
         "subquestion_and_options": normalize_subquestion(subquestion)}
    )[:24]


def _gold_letter(value: Any, *, where: str) -> str:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{where}: expected one-element answer list, got {value!r}")
    answer = str(value[0]).strip().upper()
    if answer not in "ABCDE":
        raise ValueError(f"{where}: invalid A-E answer {value!r}")
    return answer


def _task_kind(task: str) -> str:
    if task in SEQUENTIAL_TASKS:
        return "sequential"
    if task in NUMERIC_TASKS:
        return "numeric"
    return "single"


def _source_item_id(relative: str, row_index: int, subindex: int = 0) -> str:
    return f"{relative}:{row_index}:{subindex}"


def verify_snapshot(data_root: Path, *, include_herb: bool = True) -> dict[str, Any]:
    data_root = data_root.resolve()
    selected = EXPECTED_FILES if include_herb else {
        key: value for key, value in EXPECTED_FILES.items() if "herb_predict" not in key
    }
    inventory = []
    for relative, (expected_rows, expected_sha) in selected.items():
        path = data_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing official TCMBench file: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"TCMBench checksum mismatch for {relative}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        root = json.loads(path.read_text(encoding="utf-8"))
        actual_rows = len(root.get("example") or [])
        if actual_rows != expected_rows:
            raise ValueError(
                f"TCMBench row-count mismatch for {relative}: "
                f"expected {expected_rows}, got {actual_rows}"
            )
        inventory.append(
            {"path": relative, "rows": actual_rows, "sha256": actual_sha,
             "type": root.get("type")}
        )
    return {
        "data_root": str(data_root),
        "expected_data_commit": DATA_COMMIT,
        "historical_pipeline_reference_commit": PIPELINE_REFERENCE_COMMIT,
        "files": inventory,
        "snapshot_fingerprint": canonical_hash(inventory),
    }


def _read_task(data_root: Path, task: str) -> tuple[str, dict[str, Any]]:
    if task not in TASK_TO_FILE:
        raise ValueError(f"unknown task {task!r}")
    relative = TASK_TO_FILE[task]
    path = data_root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_rows, expected_sha = EXPECTED_FILES[relative]
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(f"checksum mismatch for {relative}: {actual_sha}")
    root = json.loads(path.read_text(encoding="utf-8"))
    if root.get("type") != task or len(root.get("example") or []) != expected_rows:
        raise ValueError(f"unexpected type/count in {relative}")
    return relative, root


def load_episodes(data_root: Path, tasks: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load selected tasks into model-independent episode records."""
    data_root = data_root.resolve()
    episodes: list[dict[str, Any]] = []
    task_metadata: dict[str, Any] = {}
    data_exclusions: list[dict[str, Any]] = []
    global_index = 0
    for task in tasks:
        if task == "herb_predict":
            raise ValueError("herb_predict is intentionally excluded; see inspect-data/README")
        relative, root = _read_task(data_root, task)
        kind = _task_kind(task)
        task_rows = []
        for row_index, source in enumerate(root["example"]):
            case_index = int(source.get("index", row_index))
            base = {
                "index": global_index,
                "episode_id": f"{relative}:{row_index}",
                "task": task,
                "level": relative.split("/", 1)[0],
                "relative_path": relative,
                "source_row_index": row_index,
                "case_index": case_index,
                "kind": kind,
                "prefix_prompt": str(root["prefix_prompt"]),
            }
            if kind == "sequential":
                share = str(source.get("share_content") or "").strip()
                questions = source.get("question")
                if not isinstance(questions, list) or not questions:
                    raise ValueError(f"{relative}:{row_index} lacks subquestions")
                items = []
                invalid_episode = False
                for subindex, question in enumerate(questions):
                    user_text = str(question.get("sub_question") or "").strip()
                    try:
                        gold = _gold_letter(
                            question.get("answer"), where=f"{relative}:{row_index}:{subindex}"
                        )
                    except ValueError as exc:
                        data_exclusions.append(
                            {"item_id": _source_item_id(relative, row_index, subindex),
                             "reason": str(exc)}
                        )
                        invalid_episode = True
                        break
                    items.append(
                        {
                            "item_id": _source_item_id(relative, row_index, subindex),
                            "semantic_id": semantic_id(share, user_text),
                            "presentation_index": subindex,
                            "user_text": user_text,
                            "gold": gold,
                        }
                    )
                if invalid_episode:
                    continue
                base.update({"share_content": share, "items": items})
            elif kind == "numeric":
                question = str(source.get("question") or "").strip()
                option = str(source.get("option") or "").strip()
                answer = str(source.get("answer") or "").strip()
                numbers = [int(value) for value in _NUMBERED_LINE.findall(question)]
                if not numbers or numbers != list(range(1, max(numbers) + 1)):
                    raise ValueError(f"non-contiguous numbered list at {relative}:{row_index}")
                if not answer.isdigit() or int(answer) not in numbers:
                    raise ValueError(f"invalid numeric gold at {relative}:{row_index}")
                base.update(
                    {
                        "share_content": "",
                        "option": option,
                        "option_count": max(numbers),
                        "items": [
                            {
                                "item_id": _source_item_id(relative, row_index),
                                "semantic_id": semantic_id(option, question),
                                "presentation_index": 0,
                                "user_text": question,
                                "gold": answer,
                            }
                        ],
                    }
                )
            else:
                question = str(source.get("question") or "").strip()
                try:
                    gold = _gold_letter(
                        source.get("answer"), where=f"{relative}:{row_index}"
                    )
                except ValueError as exc:
                    data_exclusions.append(
                        {"item_id": _source_item_id(relative, row_index), "reason": str(exc)}
                    )
                    continue
                base.update(
                    {
                        "share_content": "",
                        "items": [
                            {
                                "item_id": _source_item_id(relative, row_index),
                                "semantic_id": semantic_id("", question),
                                "presentation_index": 0,
                                "user_text": question,
                                "gold": gold,
                            }
                        ],
                    }
                )
            episodes.append(base)
            task_rows.append(base)
            global_index += 1
        task_metadata[task] = {
            "relative_path": relative,
            "source_rows": len(root["example"]),
            "episodes": len(task_rows),
            "items": sum(len(row["items"]) for row in task_rows),
            "kind": kind,
            "unscorable_source_items": sum(
                str(row["item_id"]).startswith(relative + ":") for row in data_exclusions
            ),
        }
    manifest_core = {
        "protocol_version": PROTOCOL_VERSION,
        "tasks": list(tasks),
        "task_metadata": task_metadata,
        "data_exclusions": data_exclusions,
        "episodes": [
            {
                "index": row["index"],
                "episode_id": row["episode_id"],
                "task": row["task"],
                "case_index": row["case_index"],
                "item_ids": [item["item_id"] for item in row["items"]],
                "semantic_ids": [item["semantic_id"] for item in row["items"]],
                "gold": [item["gold"] for item in row["items"]],
            }
            for row in episodes
        ],
    }
    manifest_core["manifest_fingerprint"] = canonical_hash(manifest_core)
    return episodes, manifest_core


def build_sdt_pair_manifest(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Join SDT variants semantically inside each case, never by position."""
    by_task_case: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for episode in episodes:
        task = str(episode["task"])
        if task in PRIMARY_VALID_TASKS:
            case = int(episode["case_index"])
            if case in by_task_case[task]:
                raise ValueError(f"duplicate {task} case index {case}")
            by_task_case[task][case] = episode
    if not all(task in by_task_case for task in PRIMARY_VALID_TASKS):
        return {
            "available": False,
            "reason": "SDT, SDT_reverse, and SDT_shuffle must be loaded together",
            "pairs": [],
        }
    expected_cases = set(range(77))
    for task in PRIMARY_VALID_TASKS:
        if set(by_task_case[task]) != expected_cases:
            raise ValueError(f"{task} does not contain exactly case indices 0..76")
    pairs = []
    cases = []
    for perturbation in ("SDT_reverse", "SDT_shuffle"):
        for case in sorted(expected_cases):
            clean = by_task_case["SDT"][case]
            perturbed = by_task_case[perturbation][case]
            if normalize_text(clean["share_content"]) != normalize_text(perturbed["share_content"]):
                raise ValueError(f"share_content mismatch for {perturbation} case {case}")
            clean_items = {item["semantic_id"]: item for item in clean["items"]}
            pert_items = {item["semantic_id"]: item for item in perturbed["items"]}
            if len(clean_items) != len(clean["items"]) or len(pert_items) != len(perturbed["items"]):
                raise ValueError(f"duplicate semantic subquestion in case {case}")
            if clean_items.keys() != pert_items.keys():
                raise ValueError(f"semantic item set mismatch for {perturbation} case {case}")
            for sem_id, clean_item in clean_items.items():
                pert_item = pert_items[sem_id]
                if clean_item["gold"] != pert_item["gold"]:
                    raise ValueError(f"gold mismatch for {perturbation}/{case}/{sem_id}")
                pairs.append(
                    {
                        "perturbation": perturbation,
                        "case_index": case,
                        "semantic_id": sem_id,
                        "gold": clean_item["gold"],
                        "clean_item_id": clean_item["item_id"],
                        "perturbed_item_id": pert_item["item_id"],
                    }
                )
            cases.append(
                {
                    "perturbation": perturbation,
                    "case_index": case,
                    "clean_episode_id": clean["episode_id"],
                    "perturbed_episode_id": perturbed["episode_id"],
                    "semantic_ids": sorted(clean_items),
                }
            )
    by_perturbation = Counter(pair["perturbation"] for pair in pairs)
    if by_perturbation != Counter({"SDT_reverse": 264, "SDT_shuffle": 264}):
        raise ValueError(f"unexpected SDT pair counts: {dict(by_perturbation)}")
    core = {
        "available": True,
        "join_key": "case_index + SHA256(normalized share_content, subquestion+options)",
        "position_join_forbidden": True,
        "clean_task": "SDT",
        "perturbations": ["SDT_reverse", "SDT_shuffle"],
        "case_pairs": len(cases),
        "question_pairs": len(pairs),
        "pairs_by_perturbation": dict(by_perturbation),
        "cases": cases,
        "pairs": pairs,
    }
    core["pair_manifest_fingerprint"] = canonical_hash(core)
    return core


def inspect_known_data_issues(data_root: Path) -> dict[str, Any]:
    """Machine-check the issues that determine the valid primary track."""
    read = lambda task: _read_task(data_root, task)[1]["example"]
    fku = read("FKU")
    ddst = read("DDST")
    ddst_hard = read("DDST_hard")
    sddt = read("SDDT")
    sddt_hard = read("SDDT_hard")
    herb_relative, herb_root = _read_task(data_root, "herb_predict")

    ddst_missing_options = sum(len(_OPTION_LINE.findall(str(row["question"]))) < 2 for row in ddst)
    hard_last_target = 0
    hard_group_map_valid = 0
    ddst_by_option: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ddst:
        ddst_by_option[str(row["option"]).strip()].append(row)
    for row in ddst_hard:
        numbered = [int(value) for value in _NUMBERED_LINE.findall(str(row["question"]))]
        answer = int(str(row["answer"]))
        hard_last_target += bool(numbered and answer == max(numbered))
        hard_group_map_valid += bool(
            str(row["option"]).strip() in ddst_by_option
            and len(ddst_by_option[str(row["option"]).strip()]) + 1 == max(numbered)
        )

    hard_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sddt_hard:
        hard_by_question[normalize_text(row["question"])].append(row)
    sddt_exact_rows = 0
    sddt_pair_instances = 0
    sddt_same_gold = 0
    for row in sddt:
        matches = hard_by_question.get(normalize_text(row["question"]), [])
        if matches:
            sddt_exact_rows += 1
        for match in matches:
            sddt_pair_instances += 1
            sddt_same_gold += row["answer"] == match["answer"]

    prompt = str(herb_root["prefix_prompt"])
    herb_questions_ask_syndrome = sum(
        "推断出证候和治法" in str(row.get("question") or "")
        for row in herb_root["example"]
    )
    return {
        "FKU": {
            "rows": len(fku),
            "rows_with_non_A_to_E_gold": sum(
                not (
                    isinstance(row.get("answer"), list)
                    and len(row["answer"]) == 1
                    and str(row["answer"][0]).upper() in "ABCDE"
                )
                for row in fku
            ),
            "handling": "unscorable rows are excluded with item IDs in task_manifest.json",
        },
        "DDST": {
            "rows": len(ddst),
            "rows_missing_visible_A_to_E_options": ddst_missing_options,
            "valid_clean_robustness_pair": False,
        },
        "DDST_hard": {
            "rows": len(ddst_hard),
            "gold_is_last_all_questions_choice": hard_last_target,
            "source_group_mapping_valid": hard_group_map_valid,
            "position_shortcut_rate": hard_last_target / len(ddst_hard),
            "valid_primary_robustness_task": False,
        },
        "SDDT_to_SDDT_hard": {
            "clean_rows": len(sddt),
            "hard_rows": len(sddt_hard),
            "clean_rows_with_identical_hard_input": sddt_exact_rows,
            "identical_input_pair_instances": sddt_pair_instances,
            "instances_with_same_gold": sddt_same_gold,
            "valid_clean_robustness_pair": False,
            "reason": "identical inputs have conflicting labels and are not perturbations",
        },
        "herb_predict": {
            "relative_path": herb_relative,
            "rows": len(herb_root["example"]),
            "prompt_requests_20_herbs": "推荐20种中药" in prompt,
            "rows_whose_embedded_question_requests_syndrome_and_therapy": herb_questions_ask_syndrome,
            "included_in_13_task_suite": False,
        },
    }


def load_overlap_exclusions(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"item_ids": set(), "manifest": None, "fingerprint": None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("excluded_item_ids") or []
    normalized = {str(value).replace("\\", "/") for value in raw}
    if len(normalized) != len(raw):
        raise ValueError("duplicate IDs in overlap exclusion manifest")
    return {
        "item_ids": normalized,
        "manifest": payload,
        "fingerprint": sha256_file(path),
    }


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [centre - half, centre + half]


def metric_block(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(items)
    correct = sum(bool(item["correct"]) for item in items)
    strict = sum(bool(item["strict_format"]) for item in items)
    parse_failures = sum(not bool(item["parse_ok"]) for item in items)
    return {
        "n": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "accuracy_wilson_95": wilson_interval(correct, total),
        "strict_format": strict,
        "strict_format_rate": strict / total if total else None,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / total if total else None,
    }


def exact_mcnemar(clean_only: int, perturbed_only: int) -> float:
    discordant = clean_only + perturbed_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(clean_only, perturbed_only) + 1))
    return min(1.0, 2 * tail / (2**discordant))
