from dataclasses import dataclass


@dataclass(frozen=True)
class Phase4ClassifiedRunner:
    name: str
    category: str
    case_ids: tuple[str, ...]
    runner_type: str = "service_runner"

    def supports(self, case_id: str) -> bool:
        return case_id in self.case_ids


class ApiRunner(Phase4ClassifiedRunner):
    pass


class ServiceRunner(Phase4ClassifiedRunner):
    pass


class RolePermissionRunner(ServiceRunner):
    pass


class MCPPermissionRunner(ServiceRunner):
    pass


class SafetyRunner(ApiRunner):
    pass


class EvidenceRunner(ServiceRunner):
    pass


class DeidentificationRunner(ServiceRunner):
    pass


class MemoryIsolationRunner(ServiceRunner):
    pass


class GraphConsistencyRunner(ServiceRunner):
    pass


class FallbackRunner(ServiceRunner):
    pass


class DemoScenarioRunner(ServiceRunner):
    pass


def phase4_runner_catalog() -> list[Phase4ClassifiedRunner]:
    return [
        RolePermissionRunner("RolePermissionRunner", "role_permission", ("P4_ROLE_001", "P4_ROLE_002", "P4_ROLE_003")),
        MCPPermissionRunner("MCPPermissionRunner", "mcp_permission", ("P4_MCP_001", "P4_MCP_002", "P4_MCP_003")),
        SafetyRunner("SafetyRunner", "safety", ("P4_SAFETY_001", "P4_SAFETY_002", "P4_SAFETY_003"), "api_runner"),
        EvidenceRunner("EvidenceRunner", "evidence", ("P4_EVIDENCE_001", "P4_EVIDENCE_002", "P4_EVIDENCE_003")),
        DeidentificationRunner("DeidentificationRunner", "deidentification", ("P4_DEID_001",)),
        MemoryIsolationRunner("MemoryIsolationRunner", "memory_isolation", ("P4_MEMORY_001",)),
        GraphConsistencyRunner("GraphConsistencyRunner", "graph_consistency", ("P4_GRAPH_001",)),
        FallbackRunner("FallbackRunner", "adapter_fallback", ("P4_FALLBACK_001",)),
        DemoScenarioRunner("DemoScenarioRunner", "demo_scenario", ("P4_DEMO_001",)),
    ]
