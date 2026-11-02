#!/usr/bin/env python3
"""Private loopback OpenAI-compatible server for the CARE Qwen3.5 model.

This intentionally uses only the Python standard library plus the already
installed torch/transformers stack.  It is a sequential evaluation server, not
a production service.  Request bodies and generated text are never logged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MAX_REQUEST_BYTES = 4 * 1024 * 1024


class ClientError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        *,
        safe_metrics: dict[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.safe_metrics = safe_metrics or {}


class CareRuntime:
    def __init__(
        self,
        *,
        model_path: str,
        served_model_name: str,
        max_model_len: int,
        seed: int,
    ) -> None:
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        if getattr(config, "model_type", None) != "qwen3_5_text":
            raise RuntimeError(
                f"expected qwen3_5_text, got {getattr(config, 'model_type', None)!r}"
            )
        configured_limit = getattr(config, "max_position_embeddings", None)
        text_config = getattr(config, "text_config", None)
        if configured_limit is None and text_config is not None:
            configured_limit = getattr(text_config, "max_position_embeddings", None)
        if max_model_len < 2048:
            raise RuntimeError("max_model_len must be at least 2048")
        if isinstance(configured_limit, int) and max_model_len > configured_limit:
            raise RuntimeError(
                f"max_model_len={max_model_len} exceeds checkpoint limit={configured_limit}"
            )
        self.torch = torch
        self.model_path = model_path
        self.served_model_name = served_model_name
        self.max_model_len = max_model_len
        self.seed = seed
        self.device = torch.device("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model.config.use_cache = True
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        self.lock = threading.Lock()
        print(
            "CARE_TRANSFORMERS_SERVER_READY_TO_BIND"
            f" transformers={transformers.__version__}"
            f" torch={torch.__version__}"
            f" cuda={torch.version.cuda}"
            f" model_type={config.model_type}",
            flush=True,
        )

    def models_response(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.served_model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local-private-eval",
                }
            ],
        }

    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        requested_model = str(payload.get("model") or "")
        if requested_model != self.served_model_name:
            raise ClientError(404, "unknown_model")
        if payload.get("stream") is True:
            raise ClientError(400, "streaming_not_supported")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ClientError(400, "messages_must_be_a_nonempty_list")
        normalized: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ClientError(400, "invalid_message")
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(
                content, str
            ):
                raise ClientError(400, "invalid_message")
            normalized.append({"role": role, "content": content})

        try:
            max_tokens = int(payload.get("max_tokens") or 512)
        except (TypeError, ValueError) as exc:
            raise ClientError(400, "invalid_max_tokens") from exc
        if max_tokens < 1 or max_tokens >= self.max_model_len:
            raise ClientError(400, "max_tokens_out_of_range")
        try:
            request_seed = int(payload.get("seed", self.seed))
        except (TypeError, ValueError) as exc:
            raise ClientError(400, "invalid_seed") from exc
        try:
            temperature = float(payload.get("temperature") or 0.0)
            top_p = float(payload.get("top_p", 1.0))
        except (TypeError, ValueError) as exc:
            raise ClientError(400, "invalid_sampling_parameters") from exc
        if temperature != 0.0:
            raise ClientError(400, "only_temperature_zero_is_supported")
        if top_p != 1.0:
            raise ClientError(400, "only_top_p_one_is_supported")

        try:
            rendered = self.tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError as exc:
            raise RuntimeError(
                "tokenizer does not support the required enable_thinking=False"
            ) from exc
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_tokens = int(encoded["input_ids"].shape[1])
        if prompt_tokens + max_tokens > self.max_model_len:
            raise ClientError(
                400,
                "context_length_exceeded",
                safe_metrics={
                    "prompt_tokens": prompt_tokens,
                    "requested_completion_tokens": max_tokens,
                    "max_model_len": self.max_model_len,
                },
            )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with self.lock:
            self.torch.manual_seed(request_seed)
            self.torch.cuda.manual_seed_all(request_seed)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
        new_tokens = generated[0, prompt_tokens:]
        completion_tokens = int(new_tokens.shape[0])
        content = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()
        if not content:
            raise RuntimeError("model returned empty content")
        finish_reason = "length" if completion_tokens >= max_tokens else "stop"
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


class PrivateServer(ThreadingHTTPServer):
    daemon_threads = True
    runtime: CareRuntime


class Handler(BaseHTTPRequestHandler):
    server: PrivateServer

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _write_json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/v1/models":
            self._write_json(200, self.server.runtime.models_response())
        elif path == "/health":
            self._write_json(200, {"status": "ok"})
        else:
            self._write_json(404, {"error": {"message": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/v1/chat/completions":
            self._write_json(404, {"error": {"message": "not_found"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            if length < 2 or length > MAX_REQUEST_BYTES:
                raise ClientError(413, "invalid_request_size")
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ClientError(400, "request_must_be_an_object")
            result = self.server.runtime.chat_completion(payload)
            usage = result.get("usage") if isinstance(result, dict) else {}
            choice = (result.get("choices") or [{}])[0]
            print(
                "GENERATION_COMPLETE"
                f" prompt_tokens={int((usage or {}).get('prompt_tokens') or 0)}"
                f" completion_tokens={int((usage or {}).get('completion_tokens') or 0)}"
                f" finish_reason={str(choice.get('finish_reason') or 'unknown')}",
                flush=True,
            )
            self._write_json(200, result)
        except ClientError as exc:
            metric_text = "".join(
                f" {key}={value}" for key, value in sorted(exc.safe_metrics.items())
            )
            print(
                f"REQUEST_REJECTED code={exc.message}{metric_text}",
                file=sys.stderr,
                flush=True,
            )
            self._write_json(exc.status, {"error": {"message": exc.message}})
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            self._write_json(400, {"error": {"message": "invalid_json"}})
        except Exception as exc:  # keep private prompts and outputs out of logs
            print(f"GENERATION_ERROR type={type(exc).__name__}", file=sys.stderr)
            self._write_json(500, {"error": {"message": "generation_failed"}})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("refusing to bind the private evaluation server externally")
    runtime = CareRuntime(
        model_path=args.model,
        served_model_name=args.served_model_name,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    server = PrivateServer((args.host, args.port), Handler)
    server.runtime = runtime
    print(
        f"CARE_TRANSFORMERS_SERVER_LISTENING host={args.host} port={args.port}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
