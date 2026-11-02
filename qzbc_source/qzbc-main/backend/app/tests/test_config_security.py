import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_production_rejects_default_jwt_secret() -> None:
    with pytest.raises(ValidationError):
        Settings(app_env="production", cors_origins=["https://guardtcm.example"])


def test_production_rejects_wildcard_cors() -> None:
    with pytest.raises(ValidationError):
        Settings(app_env="production", jwt_secret_key="a-secure-production-secret-key-value")


def test_production_accepts_explicit_security_configuration() -> None:
    settings = Settings(
        app_env="production",
        jwt_secret_key="a-secure-production-secret-key-value",
        cors_origins=["https://guardtcm.example"],
    )

    assert settings.app_env == "production"
