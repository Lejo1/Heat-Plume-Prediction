"""Result tables of the thesis from the runs in forPaper -> tables/<name>.csv and tables/<name>.tex (booktabs).

Sources per row:
 e2e runs   measurements_test{,_start}.yaml, val_eval/measurements_val{,_start}.yaml (eval_scaling_e2e.py
            --split val), scaling_test/measurements_scaling{,_start}.yaml (eval_scaling_e2e.py), measurements.yaml
 two-stage  eval_metrics.py files metrics_e2e-style_<model> <data>.yaml (keys train / val / test); the scaling
            file of export_figures (key val, datapoint 0 of the scaling dataset)
 mean       ("mean", [spec, ...]): mean +- std over repeated runs (summary rows of repeated configurations)
Missing values are written as "-", so the script can run while runs are still missing.

usage (from code/): python -m postprocessing.thesis.collect_tables"""
import csv
import statistics

import yaml

from postprocessing.thesis.common import FORPAPER, TABLES

E1 = FORPAPER / "E1_baseline/metrics"
EXPORT = FORPAPER / "figures/export/metrics"
CHAIN_PREP = "dataset_giant_100hp_varyK inputs_ixydk+s_outer outputs_t prep_with_BEST_predict_v_v4 RK45"
SIM_PREP = "dataset_giant_100hp_varyK inputs_ixydk+s_outer outputs_t"
SCALING_PREP = "giant_double_100hp_v2 inputs_ixydk+s_outer outputs_t prep_with_BEST_predict_v_v4 RK45"
TWO_STAGE = {  # label -> (val/test file, scaling file)
    "BEST_T chain": (E1 / f"metrics_e2e-style_BEST_predict_T_add_s_outer {CHAIN_PREP}.yaml",
                     EXPORT / f"metrics_e2e-style_BEST_predict_T_add_s_outer {SCALING_PREP}.yaml"),
    "BEST_T isolated": (E1 / f"metrics_e2e-style_BEST_predict_T_add_s_outer {SIM_PREP}.yaml", None),
    "sequential chain": (E1 / f"metrics_e2e-style_sequential_T_bestv {CHAIN_PREP}.yaml",
                         EXPORT / f"metrics_e2e-style_sequential_T_bestv {SCALING_PREP}.yaml"),
}


def _load(path):
    try:
        return yaml.safe_load(open(path)) if path is not None else None
    except FileNotFoundError:
        return None


def metrics(spec, source):
    """spec = (kind, ref): kind e2e | e2e_start | two; source val | test | scaling | meas -> dict or None."""
    kind, ref = spec
    if kind == "two":
        vt, sc = TWO_STAGE[ref]
        d = _load(sc if source == "scaling" else vt)
        return None if d is None else d.get("val" if source == "scaling" else source)
    run = FORPAPER / ref
    s = "_start" if kind == "e2e_start" else ""
    path = {"test": run / f"measurements_test{s}.yaml", "val": run / f"val_eval/measurements_val{s}.yaml",
            "scaling": run / f"scaling_test/measurements_scaling{s}.yaml", "meas": run / "measurements.yaml"}[source]
    return _load(path)


def value(spec, col):
    _, source, key = col
    if spec[0] == "mean":   # ("mean", [spec, ...]): mean +- std over repeats
        return mean_std([value(s, col) for s in spec[1]], key)
    if source == "meas" and spec[0] == "e2e_start":
        return None
    d = metrics(spec, source)
    return None if d is None or key not in d else d[key]


def mean_std(vals, key):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if len(vals) < 2 or key == "best_epoch":
        return None
    m = statistics.mean(vals)
    return f"{fmt(m, key)} +- {statistics.stdev(vals):.{4 if abs(m) < 1 else 2}f}"


def fmt(v, key):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    if key == "best_epoch":
        return f"{int(v)}"
    return f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}"


# column sets: (header, source, key)
T_COLS = [("val MAE", "val", "MAE [degC]"), ("test MAE", "test", "MAE [degC]"), ("test RMSE", "test", "RMSE [degC]"),
          ("test SSIM", "test", "SSIM"), ("test PAT0.1", "test", "PAT0.1 [%]")]
RUN_COLS = T_COLS[:2] + [("test traj", "test", "traj dist [cells]"), ("test vy MAE", "test", "MAE vy [m/y]"),
                         ("best epoch", "meas", "best_epoch")]
FRAME_COLS = T_COLS[:2] + [("angle hp", "test", "angle [deg] hp"), ("L_TOF hp", "test", "L_TOF hp"),
                           ("vy MAE", "test", "MAE vy [m/y]"), ("traj", "test", "traj dist [cells]"),
                           ("best epoch", "meas", "best_epoch")]
TRAJ_COLS = [("val traj", "val", "traj dist [cells]"), ("test traj", "test", "traj dist [cells]"),
             ("val MAE", "val", "MAE [degC]"), ("test MAE", "test", "MAE [degC]"), ("test vy MAE", "test", "MAE vy [m/y]"),
             ("best epoch", "meas", "best_epoch")]
SUM_COLS = T_COLS + [("scaling MAE", "scaling", "MAE [degC]"), ("test traj", "test", "traj dist [cells]")]
SCRATCH_COLS = RUN_COLS[:2] + [("scaling MAE", "scaling", "MAE [degC]")] + RUN_COLS[2:]

START = ("e2e_start", "E3_finetune/central/central_frozenT_r1")   # BEST_V + BEST_T, soft drawing (v_blur 1), no training
C, V, F = "E3_finetune/central", "E3_finetune/variants", "E4_lines/frame"
LAMN1_RUNS = ("F_lamN1", "F_lamN1_r2", "F_lamN1_r3", "F_lamN1_r4", "finetune_e2e_frame_3")   # lambda_N 1, 40 epochs, 5 repeats
TABLE_SPECS = {
    "E1_baseline": (T_COLS, [("Pelzer's chain (hard lines)", ("two", "BEST_T chain")),
                             ("isolated step 3 (simulated v)", ("two", "BEST_T isolated")),
                             ("sequential CNN2, chain", ("two", "sequential chain"))]),
    "E2_soft": (T_COLS, [("hard chain", ("two", "BEST_T chain")), ("soft, no training", START),
                         ("soft, bn_reestimate", ("e2e_start", "E3_finetune/vblur_bn/vbn_blur0_bn1"))]),
    "E3_vblur_bn": (RUN_COLS, [("start, bn_reestimate off", ("e2e_start", "E3_finetune/vblur_bn/vbn_blur0_bn0")),
                               ("start, bn_reestimate on", ("e2e_start", "E3_finetune/vblur_bn/vbn_blur0_bn1"))]
                    + [(f"v_blur {b}, bn_reestimate {'on' if n else 'off'}", ("e2e", f"E3_finetune/vblur_bn/vbn_blur{b}_bn{n}"))
                       for b in (0, 4) for n in (0, 1)]),
    "E3_central": (RUN_COLS, [("start", START)]
                   + [(f"{arm} r{r}", ("e2e", f"{C}/central_{arm}_r{r}")) for arm in ("frozenT", "joint") for r in (1, 2, 3)]),
    "E3_variants": (RUN_COLS, [(f"{n} {arm}", ("e2e", f"{V}/var_{n}_{arm}")) for n in ("directon", "freezebnv", "blur6", "repeatinner")
                               for arm in ("frozenT", "joint")]
                    + [("repeat_inner CNN1, start", ("e2e_start", f"{V}/var_repeatinner_frozenT"))]),
    "E3_from_scratch": (SCRATCH_COLS, [("start (two-stage models)", START),
                                   ("example_e2e_frame", ("e2e", "E3_finetune/from_scratch/example_e2e_frame")),
                                   ("without frame losses", ("e2e", "E3_finetune/from_scratch/scratch_noframe")),
                                   ("repo default (direct route open)", ("e2e", "E3_finetune/from_scratch/scratch_default")),
                                   ("without frame losses, CNN1 frozen", ("e2e", "E3_finetune/from_scratch/scratch_frozenV")),
                                   ("without frame losses, lambda_v 0", ("e2e", "E3_finetune/from_scratch/scratch_pureT")),
                                   ("frame losses, new CNN1", ("e2e", "E3_finetune/from_scratch/scratch_newcnn1")),
                                   ("search best by val (trial 1)", ("e2e", "E3_finetune/from_scratch/hsearch_example_e2e/trials/1")),
                                   ("search best by test (trial 6)", ("e2e", "E3_finetune/from_scratch/hsearch_example_e2e/trials/6"))]),
    "E4_trajectory": (TRAJ_COLS, [("start", ("e2e_start", "E4_lines/trajectory/traj_lr1e-5")),
                                  ("lambda_traj 3e-4, lr 1e-5", ("e2e", "E4_lines/trajectory/traj_lr1e-5")),
                                  ("lambda_traj 3e-5, lr 1e-4 -> 1e-5 (2026-10-09)", ("e2e", "E4_lines/trajectory/finetune_e2e_20261009"))]),
    "E4_frame": (FRAME_COLS, [("start", START), ("no frame losses", ("e2e", f"{F}/F_plain"))]
                 + [(f"lambda_N 1 r{i}", ("e2e", f"{F}/{r}")) for i, r in enumerate(LAMN1_RUNS, 1)]
                 + [("CNN2 alone", ("e2e", f"{F}/F_frozenV")), ("CNN2 alone r2", ("e2e", f"{F}/F_frozenV_r2")),
                    ("CNN1 alone", ("e2e", f"{F}/F_lamN1_frozenT")), ("lambda_N 0.3", ("e2e", f"{F}/F_lamN0.3")),
                    ("lambda_N 3", ("e2e", f"{F}/F_lamN3")), ("lambda_N 10", ("e2e", f"{F}/F_rep")),
                    ("lambda_TOF 1", ("e2e", f"{F}/F_lamN1_tof1")), ("lambda_TOF 3 (lambda_N 10)", ("e2e", f"{F}/F_tof3")),
                    ("heat-pump seeds", ("e2e", f"{F}/F_lamN1_hp")), ("lr warm-up", ("e2e", f"{F}/F_lamN1_warmup")),
                    ("lambda_N 1, lr 1e-5, 100 ep", ("e2e", f"{F}/finetune_e2e_frame_lr1e-5")),
                    ("lambda_N 1, 100 ep, lr drop at 25", ("e2e", f"{F}/finetune_e2e_frame_100ep"))]),
    "E4_a2": (FRAME_COLS, [("CNN2 on BEST_V lines", ("e2e", f"{F}/F_frozenV")), ("CNN2 on BEST_V lines r2", ("e2e", f"{F}/F_frozenV_r2")),
                           ("CNN2 on frame-loss CNN1 lines", ("e2e", "E4_lines/a2/A2_lamN1V_frozenV")),
                           ("CNN2 on frame-loss CNN1 lines r2", ("e2e", "E4_lines/a2/A2_lamN1V_frozenV_r2"))]),
    "E6_summary": (SUM_COLS, [("Pelzer's chain (hard)", ("two", "BEST_T chain")), ("soft, no training", START),
                              ("sequential CNN2, chain", ("two", "sequential chain")),
                              ("CNN1 alone through streamlines (3 runs)", ("mean", [("e2e", f"{C}/central_frozenT_r{r}") for r in (1, 2, 3)])),
                              ("joint T loss (3 runs)", ("mean", [("e2e", f"{C}/central_joint_r{r}") for r in (1, 2, 3)])),
                              ("CNN2 alone, lr 1e-4 (2 runs)", ("mean", [("e2e", f"{F}/{r}") for r in ("F_frozenV", "F_frozenV_r2")])),
                              ("joint + frame losses (5 runs)", ("mean", [("e2e", f"{F}/{r}") for r in LAMN1_RUNS])),
                              ("frame-loss CNN1 + CNN2 retrained (2 runs)",
                               ("mean", [("e2e", f"E4_lines/a2/{r}") for r in ("A2_lamN1V_frozenV", "A2_lamN1V_frozenV_r2")])),
                              ("from scratch + frame losses", ("e2e", "E3_finetune/from_scratch/example_e2e_frame"))]),
}
# rows whose mean +- std is appended: table -> {label of the mean row: [row labels]}
GROUPS = {"E3_central": {"frozenT mean": [f"frozenT r{r}" for r in (1, 2, 3)], "joint mean": [f"joint r{r}" for r in (1, 2, 3)]},
          "E4_frame": {"lambda_N 1 mean": [f"lambda_N 1 r{i}" for i in range(1, 6)]}}


def build(name, cols, rows):
    table = [[label] + [value(spec, c) for c in cols] for label, spec in rows]
    for mean_label, members in GROUPS.get(name, {}).items():
        sel = [r for r in table if r[0] in members]
        table.append([mean_label] + [mean_std([r[j] for r in sel], c[2]) for j, c in enumerate(cols, 1)])
    return table


def write(name, cols, table):
    header = ["configuration"] + [h for h, _, _ in cols]
    with open(TABLES / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in table:
            w.writerow([r[0]] + [fmt(v, c[2]) for v, c in zip(r[1:], cols)])
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("+-", r"$\pm$")
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join([esc(r[0])] + [esc(fmt(v, c[2])) for v, c in zip(r[1:], cols)]) + r" \\" for r in table]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n")
    missing = sum(v is None for r in table for v in r[1:])
    print(f"  {name}: {len(table)} rows, {missing} missing values -> {TABLES / name}.csv/.tex")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, (cols, rows) in TABLE_SPECS.items():
        write(name, cols, build(name, cols, rows))


if __name__ == "__main__":
    main()
