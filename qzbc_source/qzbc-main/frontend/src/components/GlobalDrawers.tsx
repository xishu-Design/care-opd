import { EvidenceDrawer } from "./evidence/EvidenceDrawer";
import { TraceDrawer } from "./trace/TraceDrawer";

export function GlobalDrawers() {
  return (
    <>
      <EvidenceDrawer />
      <TraceDrawer />
    </>
  );
}
