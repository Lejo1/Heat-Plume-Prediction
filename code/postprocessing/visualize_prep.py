"""
Quick standalone viewer for prepared .pt datapoints (no model needed).
Plots each input channel, with the streamline channels highlighted.

Usage:
    python postprocessing/visualize_prep.py "../datasets_prep/dataset_giant_100hp_varyK inputs_ixydk+s_outer outputs_t" RUN_1
    # or just pass the dataset dir to use the first available run
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-GUI backend: write PNGs instead of opening a window
import matplotlib.pyplot as plt
import torch
import yaml


def load_channel_names(dataset_dir: Path):
    """Map channel index -> property name from info.yaml (fallback to generic names)."""
    info_path = dataset_dir / "info.yaml"
    if not info_path.exists():
        return None
    with open(info_path) as f:
        info = yaml.safe_load(f)
    names = {}
    for name, stats in info["Inputs"].items():
        names[stats["index"]] = name
    return names


def visualize(dataset_dir: Path, run: str = None, save: bool = True):
    inputs_dir = dataset_dir / "Inputs"
    if run is None:
        run = sorted(p.stem for p in inputs_dir.glob("*.pt"))[0]
    run = run if run.endswith(".pt") else f"{run}.pt"

    x = torch.load(inputs_dir / run)           # shape [C, H, W]
    print(f"{run}: tensor shape {tuple(x.shape)}")

    names = load_channel_names(dataset_dir) or {i: f"channel {i}" for i in range(x.shape[0])}

    n = x.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i in range(n):
        ax = axes[i]
        title = names.get(i, f"channel {i}")
        # streamline channels are stored in [0,1]; fix scale so faint lines stay visible
        is_streamline = "streamline" in title.lower()
        im = ax.imshow(
            x[i].numpy(),
            origin="lower",
            cmap="magma" if is_streamline else "viridis",
            vmin=0 if is_streamline else None,
            vmax=1 if is_streamline else None,
            interpolation="nearest",
        )
        ax.set_title(f"[{i}] {title}", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{dataset_dir.name}  /  {run}")
    fig.tight_layout()

    if save:
        out = dataset_dir / f"preview_{Path(run).stem}.png"
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    dataset_dir = Path(sys.argv[1])
    run = sys.argv[2] if len(sys.argv) > 2 else None
    visualize(dataset_dir, run)
