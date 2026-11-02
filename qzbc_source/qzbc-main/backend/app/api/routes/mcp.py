from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_doctor
from app.core.responses import success_response
from app.db.models import User
from app.mcp.adapter_registry import AdapterRegistry
from app.mcp.audit import list_tool_calls
from app.mcp.contracts import ToolContext, ToolRequest
from app.mcp.gateway import MCPGateway
from app.mcp.registry import all_tool_specs
from app.schemas.mcp import MockToolCallRequest

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


@router.get("/tool-calls")
async def tool_calls(current_user: User = Depends(require_doctor), db: Session = Depends(get_db)) -> dict:
    del current_user
    rows = list_tool_calls(db)
    return success_response(
        [
            {
                "id": row.id,
                "user_id": row.user_id,
                "role": row.role,
                "tool_name": row.tool_name,
                "allowed": row.allowed,
                "success": row.success,
                "risk_level": row.risk_level,
                "requires_confirmation": row.requires_confirmation,
                "reason": row.reason,
                "arguments_summary": row.arguments_summary,
                "result_summary": row.result_summary,
                "error_code": row.error_code,
                "blocked_by": row.blocked_by,
                "backend": row.backend,
                "degraded": row.degraded,
                "fallback_used": row.fallback_used,
                "fallback_reason": row.fallback_reason,
                "latency_ms": row.latency_ms,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]
    )


@router.get("/tools")
async def tools(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    del current_user
    registry = AdapterRegistry(db)
    tools_data = []
    for spec in all_tool_specs():
        payload = spec.model_dump()
        payload.update(registry.tool_status(spec))
        tools_data.append(payload)
    db.commit()
    return success_response(tools_data)


@router.post("/mock-call")
async def mock_call(payload: MockToolCallRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    if payload.context:
        context = ToolContext(
            user_id=current_user.id,
            role=current_user.role,
            **{key: value for key, value in payload.context.items() if key not in {"user_id", "role"}},
        )
        result = MCPGateway(db).execute(ToolRequest(tool_name=payload.tool_name, arguments=payload.arguments, context=context))
        data = result.model_dump()
        data["result"] = result.result
        data["reason"] = result.error_code or ("allowed" if result.allowed else "denied")
        return success_response(data)
    return success_response(MCPGateway(db).call(current_user, payload.tool_name, payload.arguments))
