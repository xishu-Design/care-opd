from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import MCPToolCall


def list_tool_calls(db: Session) -> list[MCPToolCall]:
    return db.scalars(select(MCPToolCall).order_by(MCPToolCall.created_at.desc())).all()
