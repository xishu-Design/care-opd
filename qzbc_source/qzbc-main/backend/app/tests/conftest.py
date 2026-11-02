import tempfile
from collections.abc import Generator

import anyio
import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import get_db
from app.db.base import Base
from app.db import models  # noqa: F401
from app.main import app
from app.core.config import get_settings


class ASGITestClient:
    def __init__(self, app):
        self.app = app

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        async def do_request() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.request(method, url, **kwargs)

        return anyio.run(do_request)

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def patch(self, url: str, **kwargs) -> httpx.Response:
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)


@pytest.fixture(autouse=True)
def isolate_runtime_qa_index():
    settings = get_settings()
    original = settings.qa_knowledge_enabled
    settings.qa_knowledge_enabled = False
    try:
        yield
    finally:
        settings.qa_knowledge_enabled = original


@pytest.fixture()
def db_engine() -> Generator[Engine, None, None]:
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        engine = create_engine(f"sqlite:///{tmp.name}", connect_args={"check_same_thread": False}, future=True)
        Base.metadata.create_all(bind=engine)
        try:
            yield engine
        finally:
            Base.metadata.drop_all(bind=engine)
            engine.dispose()


@pytest.fixture()
def db_session(db_engine: Engine) -> Generator[Session, None, None]:
    TestingSessionLocal = sessionmaker(bind=db_engine, autoflush=False, autocommit=False, future=True)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def client(db_engine: Engine) -> Generator[ASGITestClient, None, None]:
    TestingSessionLocal = sessionmaker(bind=db_engine, autoflush=False, autocommit=False, future=True)

    async def override_get_db() -> Generator[Session, None, None]:
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield ASGITestClient(app)
    app.dependency_overrides.clear()


def register(client: ASGITestClient, username: str, role: str, display_name: str | None = None, password: str = "password123") -> dict:
    response = client.post(
        "/api/auth/register",
        json={
            "username": username,
            "password": password,
            "role": role,
            "display_name": display_name or username,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["data"]


def login(client: ASGITestClient, username: str, role: str, password: str = "password123") -> str:
    response = client.post(
        "/api/auth/login",
        json={"username": username, "password": password, "entry_role": role},
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    token = data["access_token"]
    # Test helpers complete the mandatory first-login rotation so feature tests
    # exercise normal role APIs. Password lifecycle tests call /login directly.
    if data["user"].get("must_change_password"):
        changed_password = f"{password}-changed"
        changed = client.post(
            "/api/auth/change-password",
            headers=auth_header(token),
            json={
                "current_password": password,
                "new_password": changed_password,
                "confirm_password": changed_password,
            },
        )
        assert changed.status_code == 200, changed.text
    return token


def auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def doctor_with_patient(client: ASGITestClient) -> tuple[str, dict]:
    register(client, "doctor_a", "doctor", "Doctor A")
    doctor_token = login(client, "doctor_a", "doctor")
    response = client.post(
        "/api/doctor/patients",
        headers=auth_header(doctor_token),
        json={
            "username": "patient_a",
            "password": "password123",
            "display_name": "Patient A",
            "age": 36,
            "sex": "female",
            "allergy_history": "花粉",
            "medical_history": "慢性胃炎",
            "body_state": "体质偏虚",
            "current_symptoms": "失眠",
        },
    )
    assert response.status_code == 200, response.text
    return doctor_token, response.json()["data"]


def create_reviewed_prescription(client: ASGITestClient, doctor_token: str, patient_id: int) -> int:
    create_response = client.post(
        "/api/doctor/prescriptions",
        headers=auth_header(doctor_token),
        json={
            "patient_id": patient_id,
            "title": "测试辅助拟方",
            "usage_method": "每日一剂，水煎服，遵医嘱。",
            "doctor_notes": "内部备注",
            "items": [{"herb_name": "酸枣仁", "dose": "10", "unit": "g"}],
        },
    )
    assert create_response.status_code == 200, create_response.text
    prescription_id = create_response.json()["data"]["id"]
    review_response = client.post(
        f"/api/doctor/prescriptions/{prescription_id}/review",
        headers=auth_header(doctor_token),
    )
    assert review_response.status_code == 200, review_response.text
    return prescription_id
