"""Enlarge the deployment plot without changing its archived measurements."""
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "tmm_v4"))
import gen_external_deployment as plotter


if __name__ == "__main__":
    plotter.OUT = OUT
    plotter.style()
    plotter.plot_deployment(height=1.90, bottom=0.20, top=0.85)
    print("Rendered enlarged deployment PDF/PNG/SVG from unchanged archived data.")
