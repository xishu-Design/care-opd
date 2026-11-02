from app.mcp.registry import registered_tools


ROLE_TOOL_ALLOWLIST: dict[str, set[str]] = {
    role: set(tools) for role, tools in registered_tools().items()
}
