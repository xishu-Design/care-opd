#!/usr/bin/env python3
"""Frozen HF inference: one model at a time, no server and no training.

Commands: init-config, preflight, observe, direct, generate, all.
Inference reads tasks and cached observations; gold is hashed, never parsed.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from common import (ARMS, TEXT_ARMS, PROTOCOL_VERSION, answer_messages, append_row,
                    canonical, digest, file_sha256, input_digest, load_tasks,
                    observation_messages, read_json, run_lock, unique_rows,
                    write_json_once)

DEFAULT_GENERATION = {
    "seed": 20260915, "max_new_tokens": 512, "observation_max_new_tokens": 256,
    "max_input_tokens": 6144, "max_pixels": 1024 * 28 * 28,
    "min_pixels": 256 * 28 * 28, "dtype": "bfloat16", "attention": "sdpa",
}
EVALUATION_USES = ("validation", "post_test_protocol_revision")


def configuration(path: Path) -> dict:
    config = read_json(path)
    if config.get("schema_version") != "care_opd.visual_transfer.config.v1":
        raise ValueError("unsupported config schema")
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("protocol_version mismatch: initialize a fresh V4 config and run root; do not reuse V3 results")
    if config.get("evaluation_use") not in EVALUATION_USES:
        raise ValueError("evaluation_use must be validation or post_test_protocol_revision; prior test results were inspected")
    if set(config["models"]) != {"vlm", *TEXT_ARMS}:
        raise ValueError("models must contain vlm, clean_sft, gkd_full, tropd_full, care_opd")
    for key in ("tasks_dir", "run_root"):
        config[key] = str(Path(config[key]).expanduser().resolve())
    config["models"] = {name: str(Path(value).expanduser().resolve())
                        for name, value in config["models"].items()}
    config["generation"] = {**DEFAULT_GENERATION, **config.get("generation", {})}
    generation = config["generation"]
    if set(generation) != set(DEFAULT_GENERATION):
        raise ValueError("unknown generation option")
    for key in ("max_new_tokens", "observation_max_new_tokens", "max_input_tokens", "max_pixels", "min_pixels"):
        if not isinstance(generation[key], int) or generation[key] < 1:
            raise ValueError(f"invalid generation setting {key}")
    if generation["min_pixels"] > generation["max_pixels"]:
        raise ValueError("min_pixels > max_pixels")
    if generation["dtype"] not in ("bfloat16", "float16", "float32"):
        raise ValueError("unsupported dtype")
    if generation["attention"] not in ("sdpa", "eager"):
        raise ValueError("this package supports sdpa/eager, not external flash-attn kernels")
    if Path(config["tasks_dir"]) == Path(config["run_root"]):
        raise ValueError("tasks and run roots must differ")
    return config


def source_fingerprints() -> dict:
    folder = Path(__file__).resolve().parent
    names = ("common.py", "evaluate.py", "prepare_slake.py", "score_results.py")
    return {name: file_sha256(folder / name) for name in names}


def environment() -> dict:
    import torch
    props = torch.cuda.get_device_properties(0)
    return {"python": sys.version.split()[0],
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": props.name,
            "gpu_total_memory_bytes": props.total_memory,
            "gpu_uuid": str(getattr(props, "uuid", "unavailable")), **{
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "Pillow", "accelerate")}}


def model_files(path: Path) -> list[Path]:
    if not (path / "config.json").is_file():
        raise ValueError(f"missing full checkpoint config.json: {path}")
    if (path / "adapter_config.json").exists():
        raise ValueError(f"provide a merged/full checkpoint, not a bare LoRA directory: {path}")
    weights = sorted(path.glob("*.safetensors")) or sorted(path.glob("pytorch_model*.bin"))
    if not weights:
        raise ValueError(f"no model weights: {path}")
    for file in weights:
        if file.stat().st_size < 1024:
            raise ValueError(f"empty or Git-LFS pointer weights: {file}")
    return sorted(set(weights + [p for pattern in ("*.json", "*.jinja", "*.model", "merges.txt", "vocab.txt")
                                  for p in path.glob(pattern) if p.is_file()]))


def fingerprint_model(path: Path, *, previous: dict | None = None, full: bool = False) -> dict:
    files = model_files(path)
    stat_map = {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files}
    if not full and previous and previous.get("stat") == stat_map:
        return previous
    print(f"HASH_MODEL path={path} files={len(files)}", flush=True)
    hashes = {p.name: file_sha256(p) for p in files}
    return {"sha256": digest(hashes), "files": hashes, "stat": stat_map}


def check_stack(config: dict) -> None:
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; set HF_PY to the proven GPU environment")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one physical GPU")
    if config["generation"]["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported; explicitly select float16 in a new config")
    for name, path in config["models"].items():
        cfg = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        if name == "vlm":
            if cfg.model_type != "qwen2_5_vl":
                raise ValueError("visual front end requires a native Qwen2.5-VL checkpoint")
            AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        else:
            if type(cfg) not in AutoModelForCausalLM._model_mapping:
                raise ValueError(f"{name}: config {cfg.model_type} is not a supported causal LM; use the text merged checkpoint")
            AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        print(f"MODEL_CONFIG_OK arm={name} model_type={cfg.model_type}", flush=True)
    print(f"HF_STACK_OK torch={torch.__version__} cuda={torch.version.cuda} GPU={torch.cuda.get_device_name(0)}", flush=True)


def validate_evaluation_use(config: dict, selection: dict) -> None:
    """Keep a previously inspected test set from being relabelled validation."""
    expected = {"validation": "validation", "post_test_protocol_revision": "test"}
    use = config.get("evaluation_use")
    dataset = selection.get("dataset")
    source_split = dataset.get("source_split") if isinstance(dataset, dict) else None
    if use not in expected or source_split != expected[use]:
        raise ValueError(f"evaluation_use={use!r} does not match dataset source_split={source_split!r}")


def bind_run(config: dict, tasks: list[dict], selection: dict, *, full_hash: bool = False,
             execution_environment: dict | None = None, extra_sources: tuple[str, ...] = ()) -> dict:
    validate_evaluation_use(config, selection)
    root = Path(config["run_root"])
    path = root / "run_manifest.json"
    previous = read_json(path) if path.exists() else None
    if not previous and (root / "arms").exists():
        raise ValueError("unbound results directory: choose a new run root")
    code = source_fingerprints()
    for name in extra_sources:
        if Path(name).name != name:
            raise ValueError("extra source must be a file in the experiment directory")
        code[name] = file_sha256(Path(__file__).resolve().parent / name)
    env = environment() if execution_environment is None else execution_environment
    task_root = Path(config["tasks_dir"])
    base = {"schema_version": "care_opd.visual_transfer.run.v1", "protocol_version": PROTOCOL_VERSION,
            "config": config, "tasks_sha256": selection["tasks_sha256"],
            "gold_sha256": selection["gold_sha256"],
            "selection_manifest_sha256": file_sha256(task_root / "selection_manifest.json"),
            "arms": list(ARMS), "code_sha256": code, "environment": env,
            "task_count": len(tasks), "selection_policy": "first_output_no_content_retry",
            "model_hash_policy": "sha256_initial_and_full_hash_option; size_mtime_rechecked_on_resume",
            "training": False, "scope": "frozen_visual_frontend_text_checkpoint_transfer_not_full_product"}
    base["revision_disclosure"] = {
        "prior_test_outputs_inspected": True,
        "change_scope": "shared_observation_and_answer_prompt_adapter_not_weights_or_gold",
        "old_results_policy": "preserved_not_replaced_or_selected_against_revision",
        "new_run_policy": "fresh_config_observations_and_responses_all_arms",
    }
    if previous:
        for key, value in base.items():
            if previous.get(key) != value:
                raise ValueError(f"run {key} differs; preserve old run and choose a fresh root")
    fingerprints = {}
    for name, model_path in config["models"].items():
        prev = previous.get("model_assets", {}).get(name) if previous else None
        fingerprints[name] = fingerprint_model(Path(model_path), previous=prev, full=full_hash)
        if prev and fingerprints[name]["sha256"] != prev["sha256"]:
            raise ValueError(f"checkpoint content changed for {name}")
    if previous:
        return previous
    base["model_assets"] = fingerprints
    base["model_fingerprints"] = {name: info["sha256"] for name, info in fingerprints.items()}
    base["protocol_sha256"] = digest(base)
    write_json_once(path, base)
    return base


def verify_images(tasks: list[dict]) -> None:
    from PIL import Image
    seen = {}
    for task in tasks:
        path = task["image_path"]
        if path is None:
            continue
        if path not in seen:
            seen[path] = file_sha256(path)
            with Image.open(path) as picture:
                picture.verify()
        if seen[path] != task["image_sha256"]:
            raise ValueError(f"image bytes changed: {path}")
    print(f"IMAGE_FILES_OK unique_images={len(seen)}", flush=True)


class HFGenerator:
    def __init__(self, path: str, generation: dict, *, visual: bool):
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
                                  Qwen2_5_VLForConditionalGeneration)
        self.torch, self.visual, self.generation = torch, visual, generation
        dtype = getattr(torch, generation["dtype"])
        kwargs = {"local_files_only": True, "trust_remote_code": False,
                  "torch_dtype": dtype, "attn_implementation": generation["attention"],
                  "device_map": {"": 0}, "low_cpu_mem_usage": True}
        if visual:
            self.processor = AutoProcessor.from_pretrained(
                path, local_files_only=True, trust_remote_code=False,
                min_pixels=generation["min_pixels"], max_pixels=generation["max_pixels"])
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, **kwargs)
            self.tokenizer = self.processor.tokenizer
        else:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            self.model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token_id is None:
            raise ValueError("tokenizer has neither EOS nor PAD")
        self.model.generation_config.do_sample = False
        self.model.generation_config.num_beams = 1

    def generate(self, messages: list[dict], image_path: str | None, max_tokens: int) -> dict:
        torch = self.torch
        torch.cuda.synchronize()
        start = time.perf_counter()
        if self.visual:
            from PIL import Image
            multimodal = [{"role": item["role"], "content": [{"type": "text", "text": item["content"]}]}
                          for item in messages]
            pictures = None
            if image_path:
                with Image.open(image_path) as picture:
                    pictures = [picture.convert("RGB")]
                multimodal[-1]["content"].insert(0, {"type": "image"})
            rendered = self.processor.apply_chat_template(multimodal, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[rendered], images=pictures, return_tensors="pt", padding=True)
            image_seen = "pixel_values" in inputs and inputs["pixel_values"].numel() > 0
            if bool(image_path) != image_seen:
                raise RuntimeError("pixel_values do not match image presence; refusing text-only fallback")
        else:
            if image_path is not None:
                raise ValueError("text checkpoint must not receive image paths")
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                           enable_thinking=False)
            inputs = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
            image_seen = False
        input_count = int(inputs["input_ids"].shape[1])
        if input_count > self.generation["max_input_tokens"]:
            raise RuntimeError(f"input_too_long tokens={input_count}; do not silently truncate evidence")
        cfg = getattr(self.model.config, "text_config", self.model.config)
        limit = getattr(cfg, "max_position_embeddings", None)
        if isinstance(limit, int) and input_count + max_tokens > limit:
            raise RuntimeError("input+output budget exceeds model context")
        inputs = inputs.to(self.model.device)
        torch.manual_seed(self.generation["seed"])
        torch.cuda.manual_seed_all(self.generation["seed"])
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                         max_new_tokens=max_tokens, use_cache=True,
                                         pad_token_id=self.tokenizer.pad_token_id)
        token_ids = output[0, input_count:]
        raw = self.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        stopped = bool(len(token_ids)) and int(token_ids[-1]) in eos
        torch.cuda.synchronize()
        return {"raw_text": raw, "finish_reason": "stop" if stopped or len(token_ids) < max_tokens else "length",
                "latency_seconds": time.perf_counter() - start, "prompt_tokens": input_count,
                "completion_tokens": int(len(token_ids)), "image_seen": bool(image_seen),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())}

    def close(self):
        del self.model
        self.processor = None
        self.tokenizer = None
        gc.collect()
        self.torch.cuda.empty_cache()


def run_phase(config: dict, manifest: dict, tasks: list[dict], phase: str, arm: str | None = None) -> None:
    root = Path(config["run_root"])
    observing = phase == "observe"
    direct = phase == "direct"
    label = "observations" if observing else ("vlm_direct" if direct else arm)
    if label is None or (not observing and label not in ARMS):
        raise ValueError("select a valid arm")
    model_key = "vlm" if observing or direct else label
    model_hash = manifest["model_fingerprints"][model_key]
    directory = root / "observations" if observing else root / "arms" / label
    outfile = directory / ("observations.jsonl" if observing else "responses.jsonl")
    existing = unique_rows(outfile)
    expected = {t["task_id"] for t in tasks}
    if set(existing) - expected:
        raise ValueError(f"unknown task rows in {outfile}")
    observations = {}
    if not observing and not direct:
        observations = unique_rows(root / "observations" / "observations.jsonl")
        if set(observations) != expected:
            raise RuntimeError("shared observations incomplete; run observe first")
    generator = None
    try:
        for index, task in enumerate(tasks, 1):
            observation = observations.get(task["task_id"])
            if observation is not None:
                if (observation.get("protocol_sha256") != manifest["protocol_sha256"]
                        or observation.get("model_fingerprint") != manifest["model_fingerprints"]["vlm"]
                        or observation.get("status") != "completed"):
                    raise ValueError("shared observation protocol/model/status mismatch")
                expected_input = input_digest(observation_messages(task), task["image_sha256"])
                if observation.get("input_sha256") != expected_input:
                    raise ValueError("shared observation input mismatch")
            messages = observation_messages(task) if observing else answer_messages(task, observation, direct=direct)
            image_path = task["image_path"] if observing or direct else None
            input_hash = input_digest(messages, task["image_sha256"] if observing or direct else None)
            obs_hash = digest(observation) if observation else None
            if task["task_id"] in existing:
                previous = existing[task["task_id"]]
                for key, value in (("status", "completed"), ("protocol_sha256", manifest["protocol_sha256"]),
                                   ("model_fingerprint", model_hash), ("input_sha256", input_hash),
                                   ("observation_sha256", obs_hash)):
                    if previous.get(key) != value:
                        raise ValueError(f"resume row {key} differs at {task['task_id']}")
                continue
            row = {key: task[key] for key in ("task_id", "family_id", "cluster_id", "condition")}
            row.update({"arm": label, "protocol_sha256": manifest["protocol_sha256"],
                        "model_fingerprint": model_hash, "input_sha256": input_hash,
                        "observation_sha256": obs_hash, "image_sha256": task["image_sha256"]})
            try:
                if observing and image_path is None:
                    result = {"raw_text": "未提供图像，无法获得视觉观察。", "finish_reason": "not_applicable",
                              "latency_seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
                              "image_seen": False, "peak_gpu_memory_bytes": 0}
                else:
                    if generator is None:
                        print(f"LOAD_MODEL stage={label} path={config['models'][model_key]}", flush=True)
                        generator = HFGenerator(config["models"][model_key], config["generation"],
                                                visual=observing or direct)
                    budget = config["generation"]["observation_max_new_tokens" if observing else "max_new_tokens"]
                    result = generator.generate(messages, image_path, budget)
                row.update(result)
                row["status"] = "completed"
                row["upstream_latency_seconds"] = observation["latency_seconds"] if observation else 0.0
                row["end_to_end_latency_seconds"] = row["latency_seconds"] + row["upstream_latency_seconds"]
                row["upstream_prompt_tokens"] = observation["prompt_tokens"] if observation else 0
                row["upstream_completion_tokens"] = observation["completion_tokens"] if observation else 0
                row["upstream_image_seen"] = observation["image_seen"] if observation else False
                # Empty, invalid JSON and truncated completions are still first outputs.
                append_row(outfile, row)
                print(f"PROGRESS stage={label} row={index}/{len(tasks)} task={task['task_id']} finish={row['finish_reason']}", flush=True)
            except Exception as exc:
                append_row(directory / "infrastructure_errors.jsonl", {
                    **row, "status": "infra_error", "error": f"{type(exc).__name__}: {exc}"})
                raise RuntimeError(f"infrastructure failure at {label}/{task['task_id']}; completed outputs frozen; repair environment and resume") from exc
        print(f"STAGE_COMPLETE stage={label} rows={len(tasks)}", flush=True)
    finally:
        if generator is not None:
            generator.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-config", help="write one persistent config; never overwrite it")
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--tasks-dir", required=True)
    init.add_argument("--run-root", required=True)
    init.add_argument("--evaluation-use", choices=EVALUATION_USES, required=True,
                      help="validation first; same-test repeats are disclosed post-test protocol revisions")
    for key in ("vlm", *TEXT_ARMS):
        init.add_argument("--" + key.replace("_", "-"), required=True)
    for command in ("preflight", "observe", "direct", "generate", "all"):
        child = sub.add_parser(command)
        child.add_argument("--config", required=True, type=Path)
        child.add_argument("--full-hash", action="store_true", help="rehash all checkpoint bytes instead of resume stat cache")
        if command == "generate":
            child.add_argument("--arm", choices=TEXT_ARMS, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "init-config":
        data = {"schema_version": "care_opd.visual_transfer.config.v1", "protocol_version": PROTOCOL_VERSION,
                "evaluation_use": args.evaluation_use, "tasks_dir": args.tasks_dir,
                "run_root": args.run_root, "generation": dict(DEFAULT_GENERATION),
                "models": {key: getattr(args, key) for key in ("vlm", *TEXT_ARMS)}}
        write_json_once(args.output, data)
        print(f"CONFIG_WRITTEN {args.output}")
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = configuration(args.config)
    tasks, selection = load_tasks(Path(config["tasks_dir"]))
    with run_lock(Path(config["run_root"])):
        check_stack(config)
        verify_images(tasks)
        manifest = bind_run(config, tasks, selection, full_hash=args.full_hash)
        print(f"PREFLIGHT_OK tasks={len(tasks)} protocol={manifest['protocol_sha256']}", flush=True)
        if args.command == "preflight":
            return
        if args.command == "all":
            run_phase(config, manifest, tasks, "observe")
            run_phase(config, manifest, tasks, "direct")
            for arm in TEXT_ARMS:
                run_phase(config, manifest, tasks, "generate", arm)
        else:
            run_phase(config, manifest, tasks, args.command, getattr(args, "arm", None))


if __name__ == "__main__":
    main()
