"""Paths and helpers shared by the thesis figure / table scripts.

All scripts read the runs collected in Bachelorarbeit/runs/forPaper (mirrored on pcsgs11 in
/scratch/veitjs/code/runs/forPaper; the same relative path from code/ on both machines) and write into
forPaper/figures/<name>/ and forPaper/tables/. Run them from code/, e.g. python -m postprocessing.thesis.plot_grad_flow"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR = Path(__file__).resolve().parents[2]
FORPAPER = (CODE_DIR / "../../runs/forPaper").resolve()
FIGURES = FORPAPER / "figures"
TABLES = FORPAPER / "tables"


def save(fig, out_dir: Path, name: str, dpi: int = 300):
    """Save a figure as png and pdf into out_dir and close it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_dir / name}.png/.pdf")
