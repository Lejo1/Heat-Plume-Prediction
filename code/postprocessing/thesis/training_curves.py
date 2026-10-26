"""Training curves from the TensorBoard events of the stage-2 runs (one scalar per epoch).

Figures (forPaper/figures/curves/):
  trajectory    train / val line distance of the trajectory-loss runs, start state (val_eval) as marker
  frame_angle   val angle without frame losses (F_plain) vs lambda_N 1 (all repeats)
  a2_val_huber  val Huber of CNN2 trained on the lines of the frame-loss CNN1 (A2) vs on BEST_V's lines (F_frozenV)
Runs with the same label share a colour (repeats); only the first one gets a legend entry.

usage (from code/): python -m postprocessing.thesis.training_curves"""
import yaml
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from postprocessing.thesis.common import FIGURES, FORPAPER, save

L = FORPAPER / "E4_lines"
FIGS = {
    "trajectory": {"ylabel": "line distance [cells]", "start_key": "traj dist [cells]", "curves": [
        ("lambda_traj 3e-4, lr 1e-5", L / "trajectory/traj_lr1e-5", ["val traj [cells]", "train traj [cells]"]),
        ("lambda_traj 3e-5, lr 1e-4 -> 1e-5 (2026-10-09)", L / "trajectory/finetune_e2e_20261009", ["val traj [cells]", "train traj [cells]"])]},
    "frame_angle": {"ylabel": "val angle [deg]", "curves": [
        ("no frame losses", L / "frame/F_plain", ["val angle [deg]"])] + [
        ("lambda_N 1", L / f"frame/{r}", ["val angle [deg]"])
        for r in ["F_lamN1", "F_lamN1_r2", "F_lamN1_r3", "F_lamN1_r4", "finetune_e2e_frame_3"]]},
    "a2_val_huber": {"ylabel": "val Huber (normalized T)", "curves": [
        ("CNN2 on lines of the frame-loss CNN1", L / f"a2/{r}", ["val Huber"]) for r in ["A2_lamN1V_frozenV", "A2_lamN1V_frozenV_r2"]] + [
        ("CNN2 on lines of BEST_V", L / f"frame/{r}", ["val Huber"]) for r in ["F_frozenV", "F_frozenV_r2"]]},
}
STYLES = ["-", "--"]   # per tag: val solid, train dashed


def scalars(run_dir, tag):
    events = sorted(run_dir.glob("events.out.tfevents.*"))
    assert events, f"no TensorBoard events in {run_dir}"
    acc = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0}).Reload()
    s = acc.Scalars(tag)
    return [e.step + 1 for e in s], [e.value for e in s]   # step 0 = after the first epoch; 0 = start state


def main():
    for name, spec in FIGS.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        colors, seen = {}, set()
        for label, run, tags in spec["curves"]:
            c = colors.setdefault(label, f"C{len(colors)}")
            for tag, ls in zip(tags, STYLES):
                x, y = scalars(run, tag)
                key = (label, tag)
                lab = None if key in seen else (label if len(tags) == 1 else f"{label}: {tag.split()[0]}")
                seen.add(key)
                ax.plot(x, y, ls=ls, color=c, lw=1.2 if lab else 0.8, label=lab)
            if "start_key" in spec:
                f = run / "val_eval/measurements_val_start.yaml"
                if f.exists():
                    ax.plot([0], [yaml.safe_load(f.read_text())[spec["start_key"]]], "o", color=c, ms=5)
        ax.set_xlabel("epochs trained"); ax.set_ylabel(spec["ylabel"]); ax.legend(fontsize=8)
        if "start_key" in spec:
            ax.set_title("marker: start state (val)", fontsize=9)
        fig.tight_layout()
        save(fig, FIGURES / "curves", name)


if __name__ == "__main__":
    main()
