"""Hyperparameter searches: best validation objective vs test T MAE per completed trial.

For every search folder (hsearch.db + trials/<n>/measurements_test.yaml) one scatter plot and one csv
(trial, objective = best val Huber(T) on normalized T, test T MAE, parameters). Reference: the start
state of the finetunes (val objective from the RMSE in val_eval/measurements_val_start.yaml of REFERENCE_RUN,
see T_SPAN; test from its measurements_test_start.yaml).
Caveat in the from-scratch search: trials with different depth / kernel size compute the objective on
different output areas.

usage (from code/): python -m postprocessing.thesis.search_scatter"""
import csv
import sqlite3

import yaml
import matplotlib.pyplot as plt

from postprocessing.thesis.common import FIGURES, FORPAPER, TABLES, save

SEARCHES = {"finetune": FORPAPER / "E3_finetune/hsearch_finetune/hsearch_e2e_FA",
            "from_scratch": FORPAPER / "E3_finetune/from_scratch/hsearch_example_e2e",
            "from_scratch_old": FORPAPER / "E3_finetune/from_scratch/hsearch_example_e2e_old"}
REFERENCE_RUN = FORPAPER / "E3_finetune/central/central_frozenT_r1"   # start state BEST_V + BEST_T, v_blur 1
T_SPAN = 5.0   # degC, Rescale span of T (10.6-15.6 degC)
# The objective is Huber (delta 1) on normalized T; all normalized errors are < 1 (max ~4.4 / 5 degC), so it equals
# 0.5 * MSE / span^2. The Huber [degC] of the val yaml cannot be converted (errors > 1 degC use the linear branch).


def trials(folder):
    c = sqlite3.connect(folder / "hsearch.db")
    rows = c.execute("select t.trial_id, t.number, v.value from trials t join trial_values v on t.trial_id = v.trial_id "
                     "where t.state = 'COMPLETE' order by t.number").fetchall()
    out = []
    for trial_id, number, value in rows:
        params = dict(c.execute("select param_name, param_value from trial_params where trial_id = ?", (trial_id,)).fetchall())
        f = folder / "trials" / str(number) / "measurements_test.yaml"
        test = yaml.safe_load(f.read_text())["MAE [degC]"] if f.exists() else None
        out.append({"trial": number, "objective": value, "test T MAE": test, **params})
    return out


def reference():
    try:
        v = 0.5 * yaml.safe_load((REFERENCE_RUN / "val_eval/measurements_val_start.yaml").read_text())["RMSE [degC]"] ** 2 / T_SPAN ** 2
    except FileNotFoundError:
        v = None
    t = yaml.safe_load((REFERENCE_RUN / "measurements_test_start.yaml").read_text())["MAE [degC]"]
    return v, t


def main():
    v_ref, t_ref = reference()
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, folder in SEARCHES.items():
        if not (folder / "hsearch.db").exists():
            print(f"{name}: no hsearch.db in {folder}"); continue
        rows = trials(folder)
        keys = sorted({k for r in rows for k in r} - {"trial", "objective", "test T MAE"})
        with open(TABLES / f"search_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trial", "objective", "test T MAE"] + keys)
            for r in sorted(rows, key=lambda r: r["objective"]):
                w.writerow([r["trial"], f"{r['objective']:.4e}", r["test T MAE"]] + [r.get(k) for k in keys])
        pts = [r for r in rows if r["test T MAE"] is not None]
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.scatter([r["objective"] for r in pts], [r["test T MAE"] for r in pts], s=18, color="#2a7fb8")
        best = min(pts, key=lambda r: r["objective"])
        ax.scatter([best["objective"]], [best["test T MAE"]], s=40, color="#c1440e", label=f"best by val (trial {best['trial']})")
        ax.axhline(t_ref, color="gray", ls="--", lw=1, label="soft chain, untrained, test")
        if v_ref is not None:
            ax.axvline(v_ref, color="gray", ls=":", lw=1, label="soft chain, untrained, val")
        ax.set_xlabel("best val objective (Huber, normalized T)"); ax.set_ylabel("test T MAE [degC]")
        ax.set_title(f"{name}: {len(rows)} completed trials", fontsize=10); ax.legend(fontsize=8)
        fig.tight_layout()
        save(fig, FIGURES / "searches", f"search_{name}")
        print(f"{name}: {len(rows)} complete, {len(pts)} with test metrics; best by val trial {best['trial']} "
              f"objective {best['objective']:.3e} test {best['test T MAE']:.4f}; best test {min(r['test T MAE'] for r in pts):.4f}")


if __name__ == "__main__":
    main()
