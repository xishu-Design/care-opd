from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings


LIGHTRAG_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DOCS_PATH = LIGHTRAG_ROOT / "input_docs" / "guardtcm_lightrag_docs.jsonl"
DEFAULT_REPORTS_DIR = LIGHTRAG_ROOT / "reports"
DEFAULT_INDEX_PATH_NAME = "guardtcm_lightrag_index.jsonl"


@dataclass(frozen=True)
class LightRAGConfig:
    enabled: bool
    mode: str
    working_dir: Path
    input_dir: Path
    input_docs_path: Path
    api_base_url: str
    timeout_seconds: int
    allow_fallback: bool
    reindex_on_start: bool
    api_key: str = ""

    @property
    def index_path(self) -> Path:
        return self.working_dir / DEFAULT_INDEX_PATH_NAME


def get_lightrag_config() -> LightRAGConfig:
    settings = get_settings()
    input_dir = _resolve_path(settings.lightrag_input_dir)
    return LightRAGConfig(
        enabled=settings.lightrag_enabled,
        mode=settings.lightrag_mode,
        working_dir=_resolve_path(settings.lightrag_working_dir),
        input_dir=input_dir,
        input_docs_path=input_dir / "guardtcm_lightrag_docs.jsonl",
        api_base_url=settings.lightrag_api_base_url.rstrip("/"),
        api_key=settings.lightrag_api_key,
        timeout_seconds=settings.lightrag_timeout_seconds,
        allow_fallback=settings.lightrag_allow_fallback,
        reindex_on_start=settings.lightrag_reindex_on_start,
    )


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path.cwd().parent / path if Path.cwd().name == "backend" else Path.cwd() / path
