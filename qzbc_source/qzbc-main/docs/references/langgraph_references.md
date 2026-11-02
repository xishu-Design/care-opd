# LangGraph References

GuardTCM Phase 5 uses the installed `langgraph` package as the real orchestration engine.

Reference sources considered:

- LangGraph official documentation for `StateGraph`, `START`, `END`, checkpoints, interrupt, and `Command`.
- LangGraph official GitHub repository examples for graph compilation and state passing patterns.
- FastAPI + LangGraph service structure examples for route-to-graph organization.
- FastAPI + MCP + LangGraph templates for layering MCP Gateway beneath graph nodes.
- LangGraph multi-agent examples for node and edge organization.

Project policy:

- GuardTCM only references structure and API usage patterns.
- No external business prompts, medical logic, agent policies, or model-call logic are copied.
- All tool execution remains behind GuardTCM MCP Gateway.
- Evidence Packet, Safety Guard, Agent Trace, and Harness remain GuardTCM-owned implementations.
