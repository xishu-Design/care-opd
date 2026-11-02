from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import agents, auth, content, demo, doctor, evaluation, harness, knowledge, lightrag, llm, mcp, patient, rules, student
from app.core.config import get_settings
from app.core.exceptions import install_exception_handlers
from app.db.init_db import init_db
from app.demo.schemas import PHASE11, VERSION11

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
install_exception_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(agents.router)
app.include_router(doctor.router)
app.include_router(patient.router)
app.include_router(student.router)
app.include_router(mcp.router)
app.include_router(knowledge.router)
app.include_router(rules.router)
app.include_router(harness.router)
app.include_router(llm.router)
app.include_router(content.router)
app.include_router(lightrag.router)
app.include_router(demo.router)
app.include_router(evaluation.router)

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "GuardTCM Backend", "phase": PHASE11}


@app.get("/api/version")
async def version() -> dict:
    return {"success": True, "data": {"version": VERSION11, "phase": PHASE11}, "message": ""}
