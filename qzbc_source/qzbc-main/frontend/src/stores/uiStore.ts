import { create } from "zustand";

import type { EvidencePacket } from "../api/types";

interface UiState {
  evidenceDrawerOpen: boolean;
  currentEvidencePacket: EvidencePacket | Record<string, unknown> | null;
  traceDrawerOpen: boolean;
  currentTraceId: string | null;
  openEvidence: (packet: EvidencePacket | Record<string, unknown>) => void;
  closeEvidence: () => void;
  openTrace: (traceId: string) => void;
  closeTrace: () => void;
}

export const useUiStore = create<UiState>((set) => ({
  evidenceDrawerOpen: false,
  currentEvidencePacket: null,
  traceDrawerOpen: false,
  currentTraceId: null,
  openEvidence: (packet) => set({ evidenceDrawerOpen: true, currentEvidencePacket: packet }),
  closeEvidence: () => set({ evidenceDrawerOpen: false }),
  openTrace: (traceId) => set({ traceDrawerOpen: true, currentTraceId: traceId }),
  closeTrace: () => set({ traceDrawerOpen: false }),
}));
