from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "GuardTCM Backend"
    app_env: str = "development"
    database_url: str = "sqlite:///./guardtcm_dev.db"
    jwt_secret_key: str = Field(default="guardtcm-dev-secret-key-change-in-production")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24
    cors_origins: list[str] = ["*"]
    knowledge_backend: str = "local_knowledge"
    rule_engine_backend: str = "local_rules"
    graph_backend: str = "persistent_graph"
    memory_backend: str = "persistent_memory"
    graphiti_db_path: str = "runtime/graphiti/guardtcm.db"
    graphiti_database: str = "guardtcm"
    graphiti_group_prefix: str = "guardtcm"
    graphiti_timeout_seconds: int = 90
    graphiti_ingest_chat_history: bool = True
    mem0_dir: str = "runtime/mem0/meta"
    mem0_qdrant_path: str = "runtime/mem0/qdrant"
    mem0_history_db_path: str = "runtime/mem0/history.db"
    mem0_collection_name: str = "guardtcm_memories"
    mem0_embedding_dims: int = 1024
    mem0_ingest_chat_history: bool = True
    mem0_search_top_k: int = 8
    mem0_min_relevance_score: float = 0.55
    conversation_context_enabled: bool = True
    conversation_recent_turns: int = 6
    conversation_context_max_chars: int = 6000
    conversation_question_max_chars: int = 600
    conversation_answer_max_chars: int = 1200
    allow_adapter_fallback: bool = True
    agent_orchestration_backend: str = "langgraph"
    langgraph_checkpoint_backend: str = "memory"
    langgraph_enable_interrupt: bool = True
    langgraph_trace_enabled: bool = True
    langgraph_thread_id_prefix: str = "guardtcm"
    human_confirm_timeout_minutes: int = 60
    llm_enabled: bool = True
    llm_provider: str = "template"
    llm_timeout_seconds: int = 30
    llm_max_input_tokens: int = 6000
    llm_max_output_tokens: int = 1200
    llm_temperature: float = 0.2
    llm_top_p: float = 1.0
    llm_seed: int = 42
    llm_allow_fallback: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_embedding_model: str = "bge-m3"
    ollama_num_ctx: int = 32768
    ollama_health_timeout_seconds: int = 3
    ollama_generate_timeout_seconds: int = 60
    openai_compatible_base_url: str = ""
    openai_compatible_api_key: str = ""
    openai_compatible_model: str = ""
    lightrag_enabled: bool = False
    lightrag_mode: str = "api"
    lightrag_working_dir: str = "backend/app/lightrag_integration/stores/working_dir"
    lightrag_input_dir: str = "backend/app/lightrag_integration/input_docs"
    lightrag_api_base_url: str = "http://localhost:9621"
    lightrag_api_key: str = ""
    lightrag_timeout_seconds: int = 30
    lightrag_allow_fallback: bool = True
    lightrag_reindex_on_start: bool = False
    qa_knowledge_enabled: bool = True
    qa_knowledge_index_path: str = "runtime/qa_knowledge/qa_index.db"
    qa_knowledge_candidate_limit: int = 30
    qa_knowledge_rerank_enabled: bool = True
    qa_knowledge_rerank_limit: int = 8
    allow_unverified_internal_sources: bool = False
    internal_knowledge_dir: str = "runtime/internal_knowledge"
    # Disabled by default.  This gate is only for a dedicated, loopback-only,
    # PRIVATE competition evaluation backend; normal product routes must never
    # accept caller-controlled evidence qualification.
    private_eval_evidence_control_enabled: bool = False
    private_eval_evidence_control_secret: str = ""
    private_eval_isolated_database_ack: str = ""
    # Separate full-product gate for public, anonymized teaching-case inputs.
    # Unlike private_eval_evidence_control this must not bypass retrieval,
    # Graphiti, Mem0, conversation context, or normal trace persistence.
    case_reasoning_evidence_control_enabled: bool = False
    case_reasoning_evidence_control_secret: str = ""
    case_reasoning_isolated_database_ack: str = ""

    @model_validator(mode="after")
    def validate_production_security(self) -> "Settings":
        if self.app_env.lower() != "production":
            return self
        if self.jwt_secret_key == "guardtcm-dev-secret-key-change-in-production" or len(self.jwt_secret_key) < 32:
            raise ValueError("Production requires a non-default JWT_SECRET_KEY of at least 32 characters")
        if "*" in self.cors_origins:
            raise ValueError("Production CORS_ORIGINS must list trusted origins and cannot contain '*'")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
