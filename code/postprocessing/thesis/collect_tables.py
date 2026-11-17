"""Result tables of the thesis from the runs in forPaper -> tables/<name>.csv and tables/<name>.tex (booktabs).

Sources per row:
 e2e runs   measurements_test{,_start}.yaml, val_eval/measurements_val{,_start}.yaml (eval_scaling_e2e.py
            --split val), scaling_test/measurements_scaling{,_start}.yaml (eval_scaling_e2e.py), measurements.yaml
 two-stage  eval_metrics.py files metrics_e2e-style_<model> <data>.yaml (keys train / val / test); the scaling
            file of eval_metrics.py --scaling (key val, datapoint 0 of the scaling dataset)
 oracle     E2_oracle/ in the layout of an e2e run (forPaper/scripts/oracle_eval.py)
 MoC        the same files with the MoC, in the subfolder moc_eval/ (e2e runs: moc_eval/measurements_<split>{,_start}.yaml;
            oracle: the layout of its main files; two-stage: E1_baseline/metrics/moc_eval/<same file name>)
Missing values are written as "-", so the script can run while runs are still missing.

usage (from code/): python -m postprocessing.thesis.collect_tables"""
import csv

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


MOC_KEY, MOC_DIR = "MoC [%]", "moc_eval"


def metrics(spec, source, sub=""):
    """spec = (kind, ref): kind e2e | e2e_start | two; source val | test | scaling | meas -> dict or None.
    sub: subfolder with an evaluation of the same model and data (MOC_DIR for the MoC)."""
    kind, ref = spec
    if kind == "two":
        vt, sc = TWO_STAGE[ref]
        path = sc if source == "scaling" else vt
        if sub and path is not None:
            path = E1 / sub / path.name
        d = _load(path)
        return None if d is None else d.get("val" if source == "scaling" else source)
    run = FORPAPER / ref
    s = "_start" if kind == "e2e_start" else ""
    # e2e_reeval: runs re-evaluated with the current code, test metrics in test_eval/ (first runs)
    test_rel = "test_eval/measurements_test.yaml" if kind == "e2e_reeval" else f"measurements_test{s}.yaml"
    rel = {"test": test_rel, "val": f"val_eval/measurements_val{s}.yaml", "train": f"train_eval/measurements_train{s}.yaml",
           "scaling": f"scaling_test/measurements_scaling{s}.yaml", "meas": "measurements.yaml"}[source]
    if sub:  # e2e runs: flat files in sub/; the oracle keeps the layout of its main files
        return _load(run / sub / rel.split("/")[-1]) or _load(run / sub / rel)
    return _load(run / rel)


def value(spec, col):
    _, source, key = col
    if source == "meas" and spec[0] == "e2e_start":
        return None
    d = metrics(spec, source, MOC_DIR if key == MOC_KEY else "")
    if key == MOC_KEY and (d is None or key not in d):   # runs evaluated with the MoC already: main files
        d = metrics(spec, source)
    return None if d is None or key not in d else d[key]


def fmt(v, key):
    if v is None:
        return "-"
    if key == "best_epoch":
        return f"{int(v)}"
    return f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}"


# column sets: (header, source, key)
MOC_COL = ("test MoC", "test", MOC_KEY)
T_COLS = [("val MAE", "val", "MAE [degC]"), ("test MAE", "test", "MAE [degC]"), ("test RMSE", "test", "RMSE [degC]"),
          ("test SSIM", "test", "SSIM"), ("test PAT0.1", "test", "PAT0.1 [%]"), MOC_COL]
RUN_COLS = T_COLS[:2] + [MOC_COL, ("test traj", "test", "traj dist [cells]"), ("test vy MAE", "test", "MAE vy [m/y]"),
                         ("best epoch", "meas", "best_epoch")]
# line metrics on the heat-pump lines of the test and of the training simulation (train: eval_scaling_e2e.py --split train)
FRAME_COLS = T_COLS[:2] + [MOC_COL, ("test traj", "test", "traj dist [cells]"), ("test L_N", "test", "L_N hp"),
                           ("test L_TOF", "test", "L_TOF hp"), ("train traj", "train", "traj dist [cells]"),
                           ("train L_N", "train", "L_N hp"), ("train L_TOF", "train", "L_TOF hp"),
                           ("best epoch", "meas", "best_epoch")]
TRAJ_COLS = [("train traj", "train", "traj dist [cells]"), ("val traj", "val", "traj dist [cells]"),
             ("test traj", "test", "traj dist [cells]"),
             ("val MAE", "val", "MAE [degC]"), ("test MAE", "test", "MAE [degC]"), MOC_COL,
             ("test vy MAE", "test", "MAE vy [m/y]"), ("best epoch", "meas", "best_epoch")]
E1_COLS = T_COLS + [("scaling MAE", "scaling", "MAE [degC]")]
SUM_COLS = T_COLS + [("scaling MAE", "scaling", "MAE [degC]"), ("test traj", "test", "traj dist [cells]")]
SCRATCH_COLS = RUN_COLS[:2] + [("scaling MAE", "scaling", "MAE [degC]")] + RUN_COLS[2:]
FIRST_COLS = T_COLS[:2] + [("test vx MAE", "test", "MAE vx [m/y]"), ("test vy MAE", "test", "MAE vy [m/y]"),
                           ("test traj", "test", "traj dist [cells]")]
BN_RE_COLS = [("test MAE", "test", "MAE [degC]"), ("test vx MAE", "test", "MAE vx [m/y]"),
              ("test vy MAE", "test", "MAE vy [m/y]"), ("test SSIM", "test", "SSIM"), MOC_COL,
              ("test traj", "test", "traj dist [cells]")]
BN_COLS = T_COLS[:2] + [MOC_COL, ("angle hp", "test", "angle [deg] hp"), ("test traj", "test", "traj dist [cells]"),
                        ("scaling MAE", "scaling", "MAE [degC]"), ("best epoch", "meas", "best_epoch")]
OLDS = "../olds"   # runs of an early stage of this research (runs/olds, next to forPaper)
SCAL_COLS = [("scaling MAE", "scaling", "MAE [degC]"), ("scaling RMSE", "scaling", "RMSE [degC]"),
             ("scaling SSIM", "scaling", "SSIM"), ("scaling PAT0.1", "scaling", "PAT0.1 [%]"),
             ("scaling MoC", "scaling", MOC_KEY)]

START = ("e2e_start", "E3_finetune/central/central_frozenT_r1")   # BEST_V + BEST_T, soft drawing (v_blur 1), no training
# the start state with the simulated velocity in place of CNN1's output (forPaper/scripts/oracle_eval.py)
ORACLE = ("e2e", "E2_oracle")
ORACLE_LABEL = "oracle"
START_LABEL = "soft chain, untrained"   # BEST_V + BEST_T with the soft drawing, no training
SV = r"$\sigma_v$"
# rows that are references, not candidates: not considered for the bold best value of a column (empty set: all rows)
REFERENCE_ROWS = {ORACLE_LABEL}
# rows whose label is set in bold (central setup of the temperature gradient)
BOLD_ROWS = {r"$\sigma_v$ = 1, CNN1 only", r"$\sigma_v$ = 1, joint"}
# tables with a grouped header in the .tex (the .csv keeps the full column names): name -> ([(group, number of columns)],
# short column headers); groups follow the column order, an empty group leaves the cell above blank, a group with a
# third element False gets no rule below it (a header split over the two rows, e.g. "best" / "epoch")
GROUPED_HEADERS = {
    "E4_frame": ([("validation", 1), ("test", 5), ("training", 3), ("best", 1, False)],
                 ["MAE", "MAE", "MoC", "traj", "$L_N$", r"$L_\mathrm{TOF}$", "traj", "$L_N$", r"$L_\mathrm{TOF}$",
                  "epoch"]),
    "A_batchnorm": ([("validation", 1), ("test", 4), ("scaling", 1), ("best", 1, False)],
                    ["MAE", "MAE", "MoC", "angle", "traj", "MAE", "epoch"]),
}
C, V, F = "E3_finetune/central", "E3_finetune/variants", "E4_lines/frame"
# one run per configuration: the median of its repeats (the same run on val and on test)
# BatchNorm (appendix chap:app-batchnorm, decided 2026-11-07): the statistics follow the weights, i.e. a frozen CNN keeps
# weights and statistics fixed, a trained CNN trains its weights and its statistics adapt (freeze_bn = freeze_weights).
# R: these reruns (*_ruleA); F_frozenV (CNN2 alone) already used this setting. RB: the reruns with the statistics of
# both CNNs fixed in every finetuning (*_bnrule; the central runs and F_lamN1_frozenT had it originally), appendix only
R, RB = "_ruleA", "_bnrule"
FROZEN_T, JOINT, LAMN1 = f"{C}/central_frozenT_r2{R}", f"{C}/central_joint_r3{R}", f"{F}/finetune_e2e_frame_3{R}"
TABLE_SPECS = {
    "E1_baseline": (E1_COLS, [("Pelzer's chain", ("two", "BEST_T chain")),
                              ("isolated step 3", ("two", "BEST_T isolated")),
                              ("sequential CNN2", ("two", "sequential chain"))]),
    "E2_soft": (T_COLS, [("Pelzer's chain", ("two", "BEST_T chain")), (START_LABEL, START), (ORACLE_LABEL, ORACLE)]),
    # central setup (sigma_v 1) first, then one change at a time; CNN1 only = CNN2 frozen
    "E3_gradient": (RUN_COLS, [(START_LABEL, START),
                               (f"{SV} = 1, CNN1 only", ("e2e", FROZEN_T)), (f"{SV} = 1, joint", ("e2e", JOINT))]
                    + [(f"{SV} = {b}, {lab}", ("e2e", f"{V}/var_blur{b}_{arm}{R}")) for b in (0, 4)
                       for arm, lab in (("frozenT", "CNN1 only"), ("joint", "joint"))]
                    + [(f"direct route open, {lab}", ("e2e", f"{V}/var_directon_{arm}{R}"))
                       for arm, lab in (("frozenT", "CNN1 only"), ("joint", "joint"))]),
    # reference run (CNN1 = BEST_V, CNN2 random, frame losses, truncated backpropagation), then one change each
    "E3_from_scratch": (SCRATCH_COLS, [(START_LABEL, START),
                                   ("reference (frame losses, truncated)", ("e2e", "E3_finetune/from_scratch/example_e2e_frame")),
                                   ("+ full backpropagation", ("e2e", "E3_finetune/from_scratch/scratch_fulladjoint")),
                                   ("- frame losses", ("e2e", "E3_finetune/from_scratch/scratch_noframe")),
                                   ("- frame losses, CNN1 frozen", ("e2e", "E3_finetune/from_scratch/scratch_frozenV")),
                                   (r"- frame losses, $\lambda_v = 0$", ("e2e", "E3_finetune/from_scratch/scratch_pureT")),
                                   ("new CNN1 (stage 1)", ("e2e", "E3_finetune/from_scratch/scratch_newcnn1"))]),
    "E4_trajectory": (TRAJ_COLS, [(START_LABEL, ("e2e_start", f"E4_lines/trajectory/traj_lr1e-5{R}")),
                                  ("lambda_traj 3e-4, lr 1e-5", ("e2e", f"E4_lines/trajectory/traj_lr1e-5{R}"))]),
    "E4_frame": (FRAME_COLS, [(START_LABEL, START), ("no frame losses", ("e2e", f"{F}/F_plain{R}")), ("lambda_N 1", ("e2e", LAMN1)),
                              ("CNN2 alone", ("e2e", f"{F}/F_frozenV")),
                    ("CNN1 alone", ("e2e", f"{F}/F_lamN1_frozenT{R}")),
                    ("CNN2 alone on the lines of CNN1 alone", ("e2e", f"E4_lines/a2/A2_lamN1V_frozenV{R}")),
                    ("lambda_N 0.3", ("e2e", f"{F}/F_lamN0.3{R}")),
                    ("lambda_N 3", ("e2e", f"{F}/F_lamN3{R}")), ("lambda_N 10", ("e2e", f"{F}/F_rep{R}")),
                    ("lambda_TOF 1", ("e2e", f"{F}/F_lamN1_tof1{R}")), ("lambda_TOF 3 (lambda_N 10)", ("e2e", f"{F}/F_tof3{R}")),
                    ("heat-pump seeds", ("e2e", f"{F}/F_lamN1_hp{R}")),
                    ("lambda_N 1, lr 1e-5, 100 ep", ("e2e", f"{F}/finetune_e2e_frame_lr1e-5{R}"))]),
    "E6_summary": (SUM_COLS, [("Pelzer's chain", ("two", "BEST_T chain")), (START_LABEL, START),
                              ("sequential CNN2", ("two", "sequential chain")),
                              ("CNN2 alone", ("e2e", f"{F}/F_frozenV")),
                              ("CNN1 alone through streamlines", ("e2e", FROZEN_T)),
                              ("joint T loss", ("e2e", JOINT)),
                              ("joint + frame losses", ("e2e", LAMN1)),
                              ("frame-loss CNN1 + CNN2 retrained", ("e2e", f"E4_lines/a2/A2_lamN1V_frozenV{R}")),
                              ("from scratch + frame losses", ("e2e", "E3_finetune/from_scratch/example_e2e_frame"))]),
    # BatchNorm statistics of the patch training vs re-estimated on the training field: Pelzer's models (soft chain) and
    # our own patch-trained models of an early stage (runs/olds, test files only), at the start and after 200 epochs of
    # finetuning with all statistics adapting (the two old runs differ only in bn_reestimate)
    "A_bn_reestimate": (BN_RE_COLS, [(f"{START_LABEL}, patch statistics", ("e2e_start", "E3_finetune/vblur_bn/vbn_blur0_bn0")),
                                     (f"{START_LABEL}, re-estimated", ("e2e_start", "E3_finetune/vblur_bn/vbn_blur0_bn1")),
                                     ("own models, patch statistics", ("e2e_start", f"{OLDS}/finetune_e2e_first")),
                                     ("own models, re-estimated", ("e2e_start", f"{OLDS}/finetune_e2e_bn_reestimate")),
                                     ("own models, finetuned", ("e2e", f"{OLDS}/finetune_e2e_first")),
                                     ("own models, re-estimated + finetuned", ("e2e", f"{OLDS}/finetune_e2e_bn_reestimate"))]),
    # pairs that differ only in freeze_bn: statistics follow the weights (used in the main part) vs statistics of both
    # CNNs fixed in every finetuning
    # rows: the configuration with the statistics following the weights, below it (indented) the same with fixed statistics
    "A_batchnorm": (BN_COLS, [(label, ("e2e", run)) for name, used, fixed in
                              [("CNN2 alone", f"{F}/F_frozenV", f"{F}/F_frozenV{RB}"),
                               ("CNN1 only", FROZEN_T, f"{C}/central_frozenT_r2"),
                               ("joint", JOINT, f"{C}/central_joint_r3"),
                               ("trajectory loss", f"E4_lines/trajectory/traj_lr1e-5{R}", f"E4_lines/trajectory/traj_lr1e-5{RB}"),
                               ("no frame losses", f"{F}/F_plain{R}", f"{F}/F_plain{RB}"),
                               ("lambda_N 1", LAMN1, f"{F}/finetune_e2e_frame_3{RB}"),
                               ("CNN1 alone", f"{F}/F_lamN1_frozenT{R}", f"{F}/F_lamN1_frozenT"),
                               ("CNN2 on the lines of CNN1 alone", f"E4_lines/a2/A2_lamN1V_frozenV{R}",
                                f"E4_lines/a2/A2_lamN1V_frozenV{RB}")]
                              for label, run in [(name, used), (r"\quad statistics fixed", fixed)]]),
}


# first end-to-end runs without pretraining (re-evaluated with the current code, test metrics in test_eval/)
TABLE_SPECS["E3_first_runs"] = (FIRST_COLS, [(START_LABEL, START),
    ("T loss only, no pretraining", ("e2e_reeval", "E3_finetune/first_runs/example_e2e_first")),
    (r"T loss + velocity term ($\lambda_v = 0.5$), no pretraining", ("e2e_reeval", "E3_finetune/first_runs/example_e2e_with_V_loss"))])
# scaling simulation: the configurations of the summary, scaling metrics only (replaces the scaling bar chart)
TABLE_SPECS["E5_scaling"] = (SCAL_COLS, TABLE_SPECS["E6_summary"][1])


def build(cols, rows):
    return [[label] + [value(spec, c) for c in cols] for label, spec in rows]


def write(name, cols, table):
    header = ["configuration"] + [h for h, _, _ in cols]
    with open(TABLES / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in table:
            w.writerow([r[0]] + [fmt(v, c[2]) for v, c in zip(r[1:], cols)])
    # escape only outside $...$ (math in the labels, e.g. $\sigma_v$)
    esc = lambda s: "$".join(p if i % 2 else p.replace("_", r"\_").replace("%", r"\%").replace("+-", r"$\pm$")
                             for i, p in enumerate(str(s).split("$")))
    # best value per column in bold (only in the .tex): highest SSIM, lowest everything else; not the best epoch,
    # not the REFERENCE_ROWS
    best = {}
    for j, (_, _, key) in enumerate(cols, 1):
        vals = [fmt(r[j], key) for r in table if isinstance(r[j], (int, float)) and r[0] not in REFERENCE_ROWS]
        if key != "best_epoch" and vals:
            best[j] = (max if key == "SSIM" else min)(vals, key=float)
    cell = lambda v, j, key: (r"\textbf{" + esc(fmt(v, key)) + "}") if best.get(j) == fmt(v, key) else esc(fmt(v, key))
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule"]
    if name in GROUPED_HEADERS:
        groups, short = GROUPED_HEADERS[name]
        assert sum(g[1] for g in groups) == len(cols) == len(short), f"{name}: grouped header does not match the columns"
        cells, rules, c0 = [""], [], 2
        for g, n, *rule in groups:
            cells.append(rf"\multicolumn{{{n}}}{{c}}{{{esc(g)}}}" if g else " & ".join([""] * n).strip())
            if g and (not rule or rule[0]):
                rules.append(rf"\cmidrule(lr){{{c0}-{c0 + n - 1}}}")
            c0 += n
        lines += [" & ".join(cells) + r" \\", "".join(rules)]
        header = ["configuration"] + short
    lines += [" & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    label = lambda s: (r"\textbf{" + esc(s) + "}") if s in BOLD_ROWS else esc(s)
    lines += [" & ".join([label(r[0])] + [cell(v, j, c[2]) for j, (v, c) in enumerate(zip(r[1:], cols), 1)]) + r" \\"
              for r in table]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n")
    missing = sum(v is None for r in table for v in r[1:])
    print(f"  {name}: {len(table)} rows, {missing} missing values -> {TABLES / name}.csv/.tex")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, (cols, rows) in TABLE_SPECS.items():
        write(name, cols, build(cols, rows))


if __name__ == "__main__":
    main()
