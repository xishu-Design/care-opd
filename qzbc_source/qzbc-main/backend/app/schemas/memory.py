from pydantic import BaseModel, Field


class ExplicitMemoryRequest(BaseModel):
    content: str = Field(min_length=2, max_length=500)
