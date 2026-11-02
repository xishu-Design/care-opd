"""Dependency-free contracts for frozen visual-evidence transfer.

This is a new evaluation adapter, not native visual CARE training.
Gold annotations never enter the generation-message builders.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ARMS = ("vlm_direct", "clean_sft", "gkd_full", "tropd_full", "care_opd")
TEXT_ARMS = ARMS[1:]
CONDITIONS = ("clean", "no_image", "swapped_image")
PROTOCOL_VERSION = "care_opd.visual_transfer.v4"

ANSWER_SYSTEM = """你正在完成医学图像的封闭式研究问答，不是临床诊疗。
你的任务是判断当前问题所表达的命题是否得到当前图像或观察文本的支持。
仅使用本次提供的证据，不用病例想象、历史回答或外部常识补充缺失的视觉事实。

判定规则对所有样本相同：
1. 当前证据明确支持问题中的命题，回答 YES。
2. 当前证据明确否定问题中的命题，回答 NO。否定证据也是有效证据，不等于证据不足。
3. 图像缺失，或与该问题相关的证据含糊、矛盾、不可辨认，才回答 ABSTAIN。
问题本身若含否定词，应判断完整命题的真假，不能只根据观察中的单个否定词选答案。
“能清楚辨认且不包含目标”与“看不清是否包含目标”不同：前者可以支持否定，后者不能。
观察来自冻结视觉模型，可能出错且未经医生核验；这是来源属性，不是每道题都必须弃答的条件。
在本研究任务中，可依据明确的观察内容作 YES/NO 判断，不需要额外取得临床确认。
这不表示观察是临床确证，也不允许据此生成诊断、处方或剂量。

请输出一个 JSON 对象，顶层严格为 title、summary、sections、warnings、next_actions、disclaimer。
title、summary、disclaimer 为字符串，warnings、next_actions 为字符串数组。
sections 严格有两项，每项仅含 heading、content、source_ids。
第一项按实际判定从以下三种结构中选择一种，不得输出多个候选：
YES 时：{"heading":"visual_answer","content":"YES","source_ids":["image:current"]}
NO 时：{"heading":"visual_answer","content":"NO","source_ids":["image:current"]}
ABSTAIN 时：{"heading":"visual_answer","content":"ABSTAIN","source_ids":[]}
YES 和 NO 都必须保留 image:current 引用，不能只给结论而把引用数组留空。
第二项 heading 为 context_value，content 原样抄写文本证据中的 context_value，
source_ids 始终为 ["text:context"]。此项仅为非医疗文本保持控制，不提供视觉问题的答案。
不论能否回答图像问题，都必须保留独立的文本 context_value。
不得把其他来源的内容当作图像观察，不得添加 source_id。
summary 简要说明本题的证据依据或具体不足，不用泛化拒答模板代替本题判断。
disclaimer 说明仅作研究。warnings 和 next_actions 可为空数组，不需要附加临床行动建议。
禁止 Markdown 代码块、JSON 外正文和内部推理过程。"""

OBSERVE_SYSTEM = """你是医学图像研究中的冻结视觉观察工具。
请读取实际提供的图像，保留能帮助下游理解该图像的可见事实，而不是只复述问题或给一句整体弃答。
用以下三个标签组织简洁的观察文本，没有最低字数要求：
图像概况：说明能辨认的图像类型、视角、部位或整体外观；不能辨认的项目不猜测。
问题相关观察：指出与当前问题直接相关的可见对象、位置、形态，以及有依据的存在或缺失。
可判定边界：说明具体哪些内容不能确定及其原因；清楚可见的事实仍应保留，不因局部不确定而全部丢弃。
只有在图像清晰度与覆盖范围足以判断时，才描述某对象明确不存在；看不清、未覆盖不能当作不存在。
若整张图像确实无法辨认，应如实说明，不为填满标签编造内容。
不要把模型未获医生核验当成无法描述可见图像的理由。
不要输出最终 YES、NO 或 ABSTAIN 标签，不推测患者病史、治疗或处方，不输出推理过程。
观察是可能出错的模型输出，不是医生审定结论。只返回上述观察文本。"""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path | str) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path | str) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid/truncated JSONL at {path}:{line_number}; preserve file for diagnosis") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_json_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"frozen file differs: {path}; choose a new run root")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextlib.contextmanager
def run_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".execution.lock"
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"run lock exists: {path}; verify its PID before removing a stale lock") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()) + "\n")
        yield
    finally:
        path.unlink(missing_ok=True)


def unique_rows(path: Path, key: str = "task_id") -> dict[str, dict]:
    result = {}
    if path.exists():
        for row in read_jsonl(path):
            ident = row.get(key)
            if not isinstance(ident, str) or not ident or ident in result:
                raise ValueError(f"missing/duplicate {key} in {path}")
            result[ident] = row
    return result


def load_tasks(directory: Path) -> tuple[list[dict], dict]:
    manifest = read_json(directory / "selection_manifest.json")
    for filename, key in (("tasks.jsonl", "tasks_sha256"), ("gold.private.jsonl", "gold_sha256")):
        if file_sha256(directory / filename) != manifest[key]:
            raise ValueError(f"frozen {filename} hash mismatch")
    tasks = list(unique_rows(directory / "tasks.jsonl").values())
    if not tasks:
        raise ValueError("no tasks")
    for task in tasks:
        for key in ("task_id", "family_id", "cluster_id", "question", "context_value"):
            if not isinstance(task.get(key), str) or not task[key]:
                raise ValueError(f"invalid task {key}")
        if task["condition"] not in CONDITIONS:
            raise ValueError("unknown condition")
        if any(key in task for key in ("answer", "gold", "original_answer", "reference_answer")):
            raise ValueError("gold annotation leaked into tasks.jsonl")
        image = task.get("image_path")
        if task["condition"] == "no_image":
            if image is not None or task.get("image_sha256") is not None:
                raise ValueError("no_image task contains an image")
        elif not image or not task.get("image_sha256"):
            raise ValueError("visible condition missing image")
    return tasks, manifest


def observation_messages(task: dict) -> list[dict]:
    # Whitelist inputs: no qid, counterpart label, split or condition tag.
    return [
        {"role": "system", "content": OBSERVE_SYSTEM},
        {"role": "user", "content": "当前问题：" + task["question"]},
    ]


def answer_messages(task: dict, observation: dict | None, *, direct: bool = False) -> list[dict]:
    packet = {"text:context": {"source_id": "text:context", "context_value": task["context_value"]}}
    if direct:
        packet["image:current"] = {
            "source_id": "image:current",
            "availability": "missing" if task["image_path"] is None else "available",
            "instruction": "读取随本请求附带的真实图像；没有图像时不得凭常识猜答案。",
        }
    else:
        if observation is None or observation["status"] != "completed":
            raise ValueError("completed shared observation required")
        packet["image:current"] = {
            "source_id": "image:current",
            "availability": "missing" if task["image_path"] is None else "available",
            "observation": observation["raw_text"],
            "origin": "frozen_visual_model_unverified_observation",
        }
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": canonical({
            "role": "doctor", "task_type": "visual_qa_research",
            "input_text": task["question"], "evidence_packet": packet,
        })},
    ]


def input_digest(messages: list[dict], image_sha256: str | None) -> str:
    return digest({"messages": messages, "image_sha256": image_sha256})
