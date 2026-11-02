from pydantic import BaseModel, Field


class MockToolCallRequest(BaseModel):
    tool_name: str
    arguments: dict = Field(default_factory=dict)
    context: dict = Field(default_factory=dict)


class ToolCallResponse(BaseModel):
    allowed: bool
    tool_name: str
    reason: str
    result: dict
