def execute_mock_tool(tool_name: str, arguments: dict) -> dict:
    return {
        "tool_name": tool_name,
        "mock": True,
        "arguments": arguments,
        "summary": f"Mock tool {tool_name} executed by MCP Gateway.",
    }
