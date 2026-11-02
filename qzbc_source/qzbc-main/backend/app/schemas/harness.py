from pydantic import BaseModel


class HarnessRunRequest(BaseModel):
    phase: str = "phase_11_demo_readiness"
    suite: str = "full"
    reset_demo_data: bool = True


class HarnessRunResponse(BaseModel):
    run_id: int
    phase: str | None = None
    suite: str | None = None
    total_cases: int
    passed: int
    failed: int
    pass_rate: float
    categories: dict
    metrics: dict | None = None
