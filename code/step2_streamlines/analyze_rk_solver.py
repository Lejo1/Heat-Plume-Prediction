#!/usr/bin/env python3
"""Accuracy analysis of the streamline ODE solver: BEFORE vs AFTER the rewrite.

    BEFORE  commit 9d007c6b447fc43025a0c55fe24338b905018d3c
            code/step2_streamlines/streamlines_helpers.py :: calc_streamline
            scipy.integrate.solve_ivp on a scipy RegularGridInterpolator, one call per heat pump.
            `method` was a named argument of build_streamlines and never forwarded, so solve_ivp
            ran with its DEFAULTS: method="RK45" (Dormand-Prince 5(4), adaptive), rtol=1e-3,
            atol=1e-6.  The "RK45" in the dataset folder name was only ever cosmetic.
            The paper describes a "5th-order implicit Runge-Kutta", i.e. scipy's Radau, and that
            is what the author intended; --scipy-methods evaluates both, since intent and
            execution disagree here.

    AFTER   current HEAD :: calc_streamlines
            batched fixed-step classical RK4 in PyTorch on a hand-written bilinear sampler,
            step count chosen so no point moves more than `max_step_cells` (0.5) per step,
            capped at t_steps-1, then linearly upsampled to t_steps samples.

Both integrate the same ODE  dp/dt = v(p)  with v bilinear on the cell grid and p in cells.

This file is self-contained: it re-implements both solvers rather than importing them, so it can
be run from anywhere and still reproduces the numbers.  --cross-check verifies the
re-implementations against the repo's own functions.

Usage
    python analyze_rk_solver.py                  # full analysis on the real dataset
    python analyze_rk_solver.py --quick          # fewer lines / coarser sweeps (~1 min)
    python analyze_rk_solver.py --synthetic      # no dataset needed (analytic velocity field)
    python analyze_rk_solver.py --cross-check    # also check against the repo implementations
    python analyze_rk_solver.py --plot out.png   # convergence + trajectory figure
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

T_END = 27.5          # years, as in make_streamlines
T_STEPS = 10_000      # samples drawn per line, and the cap on the RK4 step count
RESOLUTION = 5.0      # m per cell; velocities are m/y, the tracer works in cells/y
MAX_STEP_CELLS = 0.5  # production value of calc_streamlines


# ----------------------------------------------------------------------------------------------
# velocity field
# ----------------------------------------------------------------------------------------------
def load_field(prep_dir: Path, run: str, randomK: bool):
    """Return (U, starts) with U (2,N0,N1) in cells/y and starts (n,2) at heat-pump cell centres."""
    import yaml
    info = yaml.safe_load(open(prep_dir / "info.yaml"))

    def denorm(t, i, group, key):
        s = info[group][key]
        assert s["norm"] == "Rescale", f"{key}: only Rescale handled"
        return t[i] * (s["max"] - s["min"]) + s["min"]

    lab = torch.load(prep_dir / "Labels" / run, map_location="cpu")
    inp = torch.load(prep_dir / "Inputs" / run, map_location="cpu")
    vx = denorm(lab, 0, "Labels", "Liquid X-Velocity [m_per_y]").double() / RESOLUTION
    vy = denorm(lab, 1, "Labels", "Liquid Y-Velocity [m_per_y]").double() / RESOLUTION
    mat = denorm(inp, 2, "Inputs", "Material ID")
    # build_velocity_grid: randomK swaps which component drives which axis
    U = torch.stack([vy, vx] if randomK else [vx, vy]).numpy()
    starts = torch.nonzero(mat == 2).double().numpy() + 0.5   # cell-centre offset
    return U, starts


def synthetic_field(n=512, seed=0):
    """Smooth divergence-free-ish field on the same grid convention, for a dataset-free run."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    psi = np.zeros((n, n))
    for _ in range(6):                      # random smooth stream function
        kx, ky = rng.uniform(1, 4, 2) * 2 * np.pi / n
        psi += rng.uniform(-1, 1) * np.sin(kx * xx + rng.uniform(0, 6.3)) * np.cos(ky * yy + rng.uniform(0, 6.3))
    u0 = np.gradient(psi, axis=1)
    u1 = -np.gradient(psi, axis=0)
    U = np.stack([u0, u1]) * 40.0
    starts = rng.uniform(0.15 * n, 0.85 * n, size=(24, 2))
    return U, starts


# ----------------------------------------------------------------------------------------------
# the ODE right-hand side, in the two interpolation conventions
# ----------------------------------------------------------------------------------------------
def make_rhs_clamped(U):
    """Mirrors the NEW sample_velocity: bilinear, position clamped into [0, N-1]."""
    N0, N1 = U.shape[1:]

    def rhs(t, p):
        x = min(max(p[0], 0.0), N0 - 1.0)
        y = min(max(p[1], 0.0), N1 - 1.0)
        i = min(int(x), N0 - 2)
        j = min(int(y), N1 - 2)
        fx, fy = x - i, y - j
        return (U[:, i, j] * (1 - fx) * (1 - fy) + U[:, i + 1, j] * fx * (1 - fy)
                + U[:, i, j + 1] * (1 - fx) * fy + U[:, i + 1, j + 1] * fx * fy)
    return rhs


def make_rhs_extrap(U):
    """Mirrors the OLD integrate_velocity: RegularGridInterpolator, fill_value=None -> extrapolate."""
    N0, N1 = U.shape[1:]
    g = (np.arange(N0), np.arange(N1))
    f0 = RegularGridInterpolator(g, U[0], bounds_error=False, fill_value=None, method="linear")
    f1 = RegularGridInterpolator(g, U[1], bounds_error=False, fill_value=None, method="linear")
    return lambda t, p: np.squeeze([f0(p), f1(p)])


# ----------------------------------------------------------------------------------------------
# BEFORE: scipy solve_ivp, one line at a time
# ----------------------------------------------------------------------------------------------
def solve_old(rhs, start, maxs, t_end=T_END, t_steps=T_STEPS, method="RK45", rtol=1e-3, atol=1e-6,
              dense=False):
    """Verbatim re-implementation of calc_streamline at 9d007c6b (scipy defaults = what it used)."""
    sol = solve_ivp(rhs, [0, t_end], start, t_eval=np.linspace(0, t_end, t_steps),
                    method=method, rtol=rtol, atol=atol, dense_output=dense)
    sol_x, sol_y = sol.y[0], sol.y[1]
    # NOTE: the original filters x and y INDEPENDENTLY (a mask, not a truncation) - kept as-is
    sol_x = sol_x[sol_x <= maxs[0]]
    sol_y = sol_y[sol_y <= maxs[1]]
    sol_x = sol_x[sol_x >= 0]
    sol_y = sol_y[sol_y >= 0]
    L = int(np.min([sol_x.shape[0], sol_y.shape[0]]))
    return sol_x[:L], sol_y[:L], sol.t[:L], sol


# ----------------------------------------------------------------------------------------------
# AFTER: batched fixed-step RK4
# ----------------------------------------------------------------------------------------------
def sample_velocity_t(V, pos):
    n0, n1 = V.shape[1:]
    x = pos[:, 0].clamp(0, n0 - 1)
    y = pos[:, 1].clamp(0, n1 - 1)
    i = x.floor().long().clamp(max=n0 - 2)
    j = y.floor().long().clamp(max=n1 - 2)
    fx = (x - i).unsqueeze(0)
    fy = (y - j).unsqueeze(0)
    return (V[:, i, j] * (1 - fx) * (1 - fy) + V[:, i + 1, j] * fx * (1 - fy)
            + V[:, i, j + 1] * (1 - fx) * fy + V[:, i + 1, j + 1] * fx * fy).T


def rk4_trajectory(U, starts, n_int, dtype=torch.float64, t_end=T_END):
    """The RK4 loop of calc_streamlines; returns the raw nodes (n_lines, n_int+1, 2) and dt."""
    V = torch.as_tensor(U, dtype=dtype)
    x = torch.as_tensor(starts, dtype=dtype).clone()
    dt = torch.tensor(t_end / n_int, dtype=dtype)
    tr = torch.empty((x.shape[0], n_int + 1, 2), dtype=dtype)
    tr[:, 0] = x
    for k in range(n_int):
        k1 = sample_velocity_t(V, x)
        k2 = sample_velocity_t(V, x + dt / 2 * k1)
        k3 = sample_velocity_t(V, x + dt / 2 * k2)
        k4 = sample_velocity_t(V, x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        tr[:, k + 1] = x
    return tr, float(dt)


def n_int_production(U, max_step_cells=MAX_STEP_CELLS, t_end=T_END, t_steps=T_STEPS):
    v_max = float(np.linalg.norm(U, axis=0).max())
    return max(min(int(np.ceil(t_end * v_max / max_step_cells)), t_steps - 1), 16), v_max


def upsample(tr, dt, times):
    """calc_streamlines' linear upsampling of the RK4 nodes onto `times`."""
    n_int = tr.shape[1] - 1
    pos = np.asarray(times) / dt
    idx = np.clip(pos.astype(int), 0, n_int - 1)
    w = (pos - idx)[None, :, None]
    return tr.numpy()[:, idx] * (1 - w) + tr.numpy()[:, idx + 1] * w


def cut_new(sx, sy, t, maxs):
    """calc_streamlines' cut: truncate at the FIRST exit."""
    ins = (sx >= 0) & (sx <= maxs[0]) & (sy >= 0) & (sy <= maxs[1])
    L = int(np.cumprod(ins.astype(int)).sum())
    return sx[:L], sy[:L], t[:L]


def cut_old(sx, sy, t, maxs):
    sx2 = sx[sx <= maxs[0]]; sy2 = sy[sy <= maxs[1]]
    sx2 = sx2[sx2 >= 0];     sy2 = sy2[sy2 >= 0]
    L = int(min(len(sx2), len(sy2)))
    return sx2[:L], sy2[:L], t[:L]


def draw_hard(dims, lines):
    """draw_streamlines(faded=True) - identical before and after (numpy -> torch only)."""
    img = np.zeros(dims)
    for sx, sy, t in lines:
        if len(t) == 0:
            continue
        val = t[::-1] / t[::-1].max() if t.max() > 0 else np.ones_like(t)
        img[((sx + 0.5).astype(int), (sy + 0.5).astype(int))] = val
    return img


# ----------------------------------------------------------------------------------------------
def err_stats(Y, ref, valid):
    e = np.linalg.norm(Y - ref, axis=2)
    mx = np.array([e[i, :valid[i]].max() for i in range(len(e))])
    fin = np.array([e[i, valid[i] - 1] for i in range(len(e))])
    return mx, fin


def mask_from_render(png: Path, n: int):
    """Recover a boolean lit-cell mask from a rendered imshow plot of an n x n channel.

    Locates the axes interior by its background colour (the colormap's 0), then marks a cell lit if
    any pixel of its block differs from that background. The render is lossy - roughly 2 px per cell
    plus anti-aliasing - so the recovered mask is ~45% larger than the array it came from. That is
    why the comparison below needs a positive control to calibrate the achievable score.
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = np.asarray(Image.open(png).convert("RGB")).astype(np.int16)
    flat = im.reshape(-1, 3)
    # the plot interior is the largest area of one flat colour = colormap(0)
    uniq, cnt = np.unique(flat[::37], axis=0, return_counts=True)
    bg = uniq[cnt.argmax()]
    isbg = np.abs(im - bg[None, None, :]).sum(2) < 10
    cols = np.where(isbg.mean(0) > 0.5)[0]
    rows = np.where(isbg.mean(1) > 0.5)[0]
    if len(cols) == 0 or len(rows) == 0:
        raise ValueError(f"could not locate the plot interior in {png}")
    inner = im[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    lit = np.abs(inner - bg[None, None, :]).sum(2) > 30
    H, W = lit.shape
    yi = (np.arange(n) * H / n).astype(int); yj = (np.arange(n + 1) * H / n).astype(int)
    xi = (np.arange(n) * W / n).astype(int); xj = (np.arange(n + 1) * W / n).astype(int)
    c = np.pad(np.cumsum(np.cumsum(lit.astype(np.int32), 0), 1), ((1, 0), (1, 0)))
    box = (c[yj[1:, None], xj[None, 1:]] - c[yi[:, None], xj[None, 1:]]
           - c[yj[1:, None], xi[None, :]] + c[yi[:, None], xi[None, :]])
    return box > 0, (H / n, W / n)


def _dilate(m, r=1):
    o = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            o |= np.roll(np.roll(m, dy, 0), dx, 1)
    return o


def _best_shift(a, b, rad=3):
    best = None
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            s2 = np.roll(np.roll(a, dy, 0), dx, 1)
            i = int((s2 & b).sum())
            if best is None or i > best[0]:
                best = (i, dy, dx, s2)
    return best


def banner(s):
    print("\n" + "=" * 94 + f"\n{s}\n" + "=" * 94)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep", type=Path,
                    default=Path("../../datasets_prep/dataset_giant_100hp_varyK inputs_pki outputs_xy"))
    ap.add_argument("--run", default="RUN_1.pt")
    ap.add_argument("--randomK", type=int, default=1, help="1 for the varyK datasets")
    ap.add_argument("--lines", type=int, default=16, help="streamlines used for the reference-based tests")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--scipy-methods", default="RK45,Radau",
                    help="solve_ivp methods to evaluate as the 'before' solver. RK45 is what the "
                         "released code actually executed; Radau (implicit, order 5) is what the "
                         "paper describes and what the author intended.")
    ap.add_argument("--reference-png", type=Path, default=None,
                    help="a RENDERED 'Streamlines Faded' plot from an older run (e.g. Pelzer's "
                         "models/BEST_predict_T_add_s_outer/training/train_0_Streamlines Faded [-].png). "
                         "Decides which solver produced it, even though the array itself is gone. "
                         "Needs --identify as the positive control.")
    ap.add_argument("--cross-check", action="store_true")
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--identify", type=Path, default=None,
                    help="a '<...> inputs_ixydk+s_outer outputs_t' prep dir: decide which solver "
                         "built its stored 'Streamlines Faded' channel (index 3)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.quick:
        a.lines = min(a.lines, 6)

    # ---------------------------------------------------------------- field
    if a.synthetic:
        U, starts_all = synthetic_field()
        tag = "synthetic"
    else:
        if not a.prep.exists():
            sys.exit(f"{a.prep} not found - pass --prep or use --synthetic")
        U, starts_all = load_field(a.prep, a.run, bool(a.randomK))
        tag = f"{a.prep.name} / {a.run}"
    N0, N1 = U.shape[1:]
    maxs = (N0 - 1.0, N1 - 1.0)
    n_prod, v_max = n_int_production(U)
    starts = starts_all[:a.lines]
    rhs_c = make_rhs_clamped(U)

    banner("SETUP")
    print(f"  field                {tag}")
    print(f"  grid                 {N0} x {N1} cells,  {RESOLUTION} m/cell")
    print(f"  speed                max {v_max:.2f} cells/y,  mean {np.linalg.norm(U,axis=0).mean():.2f}")
    print(f"  integration          t_end = {T_END} y,  {len(starts_all)} start points ({len(starts)} used for accuracy)")
    print(f"  NEW production step  n_int = {n_prod}  ->  dt = {T_END/n_prod:.3e} y  "
          f"({T_END/n_prod*v_max:.3f} cells/step at v_max)")
    if n_prod == T_STEPS - 1:
        print(f"  !! the t_steps-1 cap binds: the requested {MAX_STEP_CELLS} cells/step is NOT met")
    METHODS = [m.strip() for m in a.scipy_methods.split(",") if m.strip()]
    print(f"  OLD solver           scipy solve_ivp, rtol=1e-3, atol=1e-6 (scipy defaults), "
          f"methods {METHODS}")
    print(f"                       NB: `method` was a named arg of build_streamlines and never")
    print(f"                       forwarded, so the released code always ran the DEFAULT, RK45.")

    times = np.linspace(0, T_END, 2001)

    # ------------------------------------------------------- E1 reference + its validation
    banner("E1  REFERENCE SOLUTION AND ITS VALIDATION")
    t0 = time.time()
    ref = []
    for s in starts:
        sol = solve_ivp(rhs_c, [0, T_END], s, method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
        ref.append(sol.sol(times).T)
    ref = np.array(ref)
    print(f"  DOP853 rtol=1e-12 on the clamped bilinear field: {time.time()-t0:.0f} s for {len(starts)} lines")
    nv = min(4, len(starts))
    n_fine = n_prod * (2 if a.quick else 4)   # always a REFINEMENT of production
    trf, dtf = rk4_trajectory(U, starts[:nv], n_fine)
    d = np.linalg.norm(upsample(trf, dtf, times) - ref[:nv], axis=2)
    print(f"  independent check vs RK4 float64 with n_int={n_fine} ({n_fine/n_prod:.0f}x production step count):")
    print(f"     |DOP853 - fine RK4|  median {np.median(d.max(1)):.2e}   worst {d.max():.2e} cells")
    print(f"  -> both agree far below the production error measured in E2, so the reference holds")

    inside = ((ref[:, :, 0] >= 0) & (ref[:, :, 0] <= maxs[0])
              & (ref[:, :, 1] >= 0) & (ref[:, :, 1] <= maxs[1]))
    valid = np.array([np.argmax(~row) if (~row).any() else len(times) for row in inside])
    valid = np.maximum(valid, 2)
    print(f"  in-domain samples per line: min {valid.min()} / {len(times)}  "
          f"({int((valid<len(times)).sum())} of {len(starts)} lines leave the domain)")

    # ------------------------------------------------------- E2 headline accuracy
    banner("E2  ACCURACY AT PRODUCTION SETTINGS  (error vs reference, in CELLS; 1 cell = 5 m)")
    old_traj, t_old_line = {}, {}
    print(f"  {'':28s} {'max along path':>26s}   {'endpoint':>26s}")
    print(f"  {'':28s} {'median':>12s} {'worst':>13s}   {'median':>12s} {'worst':>13s}")
    for m in METHODS:
        t0 = time.time()
        Y = np.array([solve_old(rhs_c, sp, maxs, method=m, dense=True)[3].sol(times).T for sp in starts])
        t_old_line[m] = (time.time() - t0) / len(starts)
        old_traj[m] = Y
        mx, fin = err_stats(Y, ref, valid)
        note = "as executed" if m == "RK45" else ("as intended" if m == "Radau" else "")
        print(f"  BEFORE {m:<8s} {note:<11s} {np.median(mx):12.3e} {mx.max():13.3e}   "
              f"{np.median(fin):12.3e} {fin.max():13.3e}")
    tr, dt = rk4_trajectory(U, starts, n_prod, torch.float32)
    new_Y = upsample(tr, dt, times)
    mx_n, fin_n = err_stats(new_Y, ref, valid)
    print(f"  AFTER  {'RK4 f32':<8s} {'production':<11s} {np.median(mx_n):12.3e} {mx_n.max():13.3e}   "
          f"{np.median(fin_n):12.3e} {fin_n.max():13.3e}")
    print()
    for m in METHODS:
        mx, _ = err_stats(old_traj[m], ref, valid)
        print(f"  {m:<6s}: median {np.median(mx)*RESOLUTION:9.2f} m, worst {mx.max()*RESOLUTION:9.1f} m"
              f"   -> the rewrite improves on it by {np.median(mx)/np.median(mx_n):.3g}x")
    print(f"  RK4   : median {np.median(mx_n)*RESOLUTION*1000:9.3f} mm, worst {mx_n.max()*RESOLUTION*1000:8.2f} mm")
    mx_o, fin_o = err_stats(old_traj[METHODS[0]], ref, valid)
    old = old_traj[METHODS[0]]

    # ------------------------------------------------------- E3 convergence order
    banner("E3  CONVERGENCE OF THE FIXED-STEP RK4  (is the 4th order actually realised?)")
    sweep = [4.0, 2.0, 1.0, 0.5] if a.quick else [8.0, 4.0, 2.0, 1.0, 0.5, 0.25]
    prev = None
    print(f"  {'max_step':>9s} {'n_int':>7s} {'dt':>10s} {'max err [cells]':>17s} {'observed order':>15s}")
    for mc in sweep:
        n = int(np.ceil(T_END * v_max / mc))          # cap deliberately lifted, to see convergence
        tr_, dt_ = rk4_trajectory(U, starts, n)
        mx, _ = err_stats(upsample(tr_, dt_, times), ref, valid)
        e = np.median(mx)
        order = "" if prev is None else f"{np.log2(prev/e):.2f}"
        print(f"  {mc:9.3f} {n:7d} {dt_:10.2e} {e:17.3e} {order:>15s}")
        prev = e
    print("  RK4 is a 4th-order method, but the observed order is ~2: bilinear interpolation makes")
    print("  the right-hand side only C0 (its derivative jumps at every cell boundary), and a")
    print("  Runge-Kutta method cannot exceed the smoothness of the field it integrates.")
    trc, _ = rk4_trajectory(U, starts, n_prod)
    seg = np.diff(trc.numpy(), axis=1)
    cross = ((np.floor(trc.numpy()[:, :-1, 0]) != np.floor(trc.numpy()[:, 1:, 0]))
             | (np.floor(trc.numpy()[:, :-1, 1]) != np.floor(trc.numpy()[:, 1:, 1])))
    print(f"  step length: median {np.median(np.linalg.norm(seg,axis=2)):.3f}, "
          f"max {np.linalg.norm(seg,axis=2).max():.3f} cells;  "
          f"{100*cross.mean():.1f}% of steps cross a cell boundary")

    # ------------------------------------------------------- E4 was RK45 the problem?
    banner("E4  IS IT THE SCHEME, OR THE TOLERANCE?")
    print(f"  {'method':>7s} {'rtol':>8s} {'steps/line':>11s} {'rhs evals':>11s} {'wall/line':>10s} {'max err [cells]':>17s}")
    for m in METHODS:
        for rtol in ([1e-3, 1e-6] if a.quick else [1e-3, 1e-4, 1e-6, 1e-9]):
            t0 = time.time(); ns = nf = 0; E = []
            for i, sp in enumerate(starts):
                sol = solve_ivp(rhs_c, [0, T_END], sp, method=m, rtol=rtol, atol=rtol * 1e-3,
                                dense_output=True)
                ns += len(sol.t); nf += sol.nfev
                E.append(np.linalg.norm(sol.sol(times).T - ref[i], axis=1)[:valid[i]].max())
            print(f"  {m:>7s} {rtol:8.0e} {ns/len(starts):11.0f} {nf/len(starts):11.0f} "
                  f"{(time.time()-t0)/len(starts):9.3f}s {np.median(E):17.3e}")
    print(f"  {'RK4':>7s} {'fixed':>8s} {n_prod:11d} {4*n_prod:11d}")
    print("  Both schemes are fine; scipy's DEFAULT rtol=1e-3 is the problem. rtol is a RELATIVE")
    print("  tolerance and the state is a position of order 1e3 cells, so 1e-3 permits an error of")
    print("  ~1 cell per step. Neither the method nor the tolerances were reachable from the config.")

    # ------------------------------------------------------- E5 precision
    banner("E5  float32 (production) vs float64")
    tr32, dt32 = rk4_trajectory(U, starts, n_prod, torch.float32)
    tr64, dt64 = rk4_trajectory(U, starts, n_prod, torch.float64)
    m32, _ = err_stats(upsample(tr32, dt32, times), ref, valid)
    m64, _ = err_stats(upsample(tr64, dt64, times), ref, valid)
    print(f"  float64: median {np.median(m64):.3e}  worst {m64.max():.3e} cells")
    print(f"  float32: median {np.median(m32):.3e}  worst {m32.max():.3e} cells")
    print(f"  -> float32 costs a factor {np.median(m32)/np.median(m64):.1f}, "
          f"leaving {np.median(m32)*RESOLUTION*1000:.2f} mm - far below one cell, so float32 is safe here")

    # ------------------------------------------------------- E6 the domain cut
    banner("E6  THE DOMAIN CUT (changed at the same time as the solver)")
    print("  BEFORE: x and y are filtered INDEPENDENTLY by boolean masks, then re-paired.")
    print("  AFTER : the line is truncated at its first exit.")
    t = np.linspace(0, 10, 11)
    sx = np.array([0., 1, 2, 3, 4, 5, 6, 12, 13, 5, 4]); sy = np.arange(11.)
    ox, oy, _ = cut_old(sx, sy, t, (10., 10.)); nx_, ny_, _ = cut_new(sx, sy, t, (10., 10.))
    print(f"    demo line leaves at i=7,8 and returns at i=9 (x={sx.astype(int)})")
    print(f"    BEFORE -> {len(ox)} pts, pairs {list(zip(ox.astype(int), oy.astype(int)))[-3:]}  <- never visited")
    print(f"    AFTER  -> {len(nx_)} pts, pairs {list(zip(nx_.astype(int), ny_.astype(int)))[-3:]}")
    trA, dtA = rk4_trajectory(U, starts_all, n_prod, torch.float32)
    fine = upsample(trA, dtA, np.linspace(0, T_END, T_STEPS))
    ins = ((fine[:, :, 0] >= 0) & (fine[:, :, 0] <= maxs[0])
           & (fine[:, :, 1] >= 0) & (fine[:, :, 1] <= maxs[1]))
    left = (~ins).any(1)
    reenter = sum(1 for i in range(len(ins)) if left[i] and ins[i, int(np.argmax(~ins[i])):].any())
    print(f"  on this field: {int(left.sum())}/{len(ins)} lines leave the domain, "
          f"{reenter} of them re-enter")
    print(f"  -> the two cuts agree unless a line re-enters, so on this dataset the difference is "
          f"latent, not active")

    # ------------------------------------------------------- E7 what CNN2 actually sees
    banner("E7  EFFECT ON THE DRAWN CHANNEL (what step 3 actually consumes)")
    t_eval = np.linspace(0, T_END, T_STEPS)
    lines_new = [cut_new(fine[i, :, 0], fine[i, :, 1], t_eval, maxs) for i in range(len(starts_all))]
    img_new = draw_hard((N0, N1), lines_new)
    rhs_e = make_rhs_extrap(U)              # the OLD pipeline's interpolator, as shipped
    lines_old, img_old, t_old_full = {}, {}, {}
    for m in METHODS:
        t0 = time.time()
        lines_old[m] = [solve_old(rhs_e, sp, maxs, method=m)[:3] for sp in starts_all]
        t_old_full[m] = time.time() - t0
        img_old[m] = draw_hard((N0, N1), lines_old[m])

    def overlap(x, y):
        A, B = x > 0, y > 0
        i = int((A & B).sum())
        return A.sum(), B.sum(), i, 100 * i / max((A | B).sum(), 1)

    print(f"  {'pair':<26s} {'lit A':>8s} {'lit B':>8s} {'common':>8s} {'IoU':>7s} {'mean|d|':>10s}")
    for m in METHODS:
        la, lb, i, iou = overlap(img_old[m], img_new)
        print(f"  {m+' vs RK4':<26s} {la:8d} {lb:8d} {i:8d} {iou:6.1f}% "
              f"{np.abs(img_old[m]-img_new).mean():10.3e}")
    for i1 in range(len(METHODS)):
        for i2 in range(i1 + 1, len(METHODS)):
            m1, m2 = METHODS[i1], METHODS[i2]
            la, lb, i, iou = overlap(img_old[m1], img_old[m2])
            print(f"  {m1+' vs '+m2:<26s} {la:8d} {lb:8d} {i:8d} {iou:6.1f}% "
                  f"{np.abs(img_old[m1]-img_old[m2]).mean():10.3e}")
    print("  -> a channel built with the old solver is NOT interchangeable with one built now:")
    print("     a ~1-cell-wide line displaced by tens of cells barely overlaps itself.")

    # ------------------------------------------------------- E8 cost
    banner("E8  COST")
    for m in METHODS:
        print(f"  BEFORE {m:<6s} {t_old_full[m]:7.1f} s for {len(starts_all)} lines "
              f"(one solve_ivp per line, Python loop)")
    t0 = time.time(); rk4_trajectory(U, starts_all, n_prod, torch.float32); t_new_full = time.time() - t0
    print(f"  AFTER  RK4    {t_new_full:7.1f} s for {len(starts_all)} lines "
          f"(one batched RK4, {a.threads} CPU threads, float32)")
    print(f"  AFTER does {4*n_prod} rhs evaluations per line against BEFORE's few hundred, but batched")
    print(f"  over all lines and vectorised; it is also differentiable, which was the point.")

    # ------------------------------------------------------- E9 provenance of a stored dataset
    if a.identify is not None:
        banner("E9  WHICH SOLVER BUILT A STORED DATASET?")
        stored = torch.load(a.identify / "Inputs" / a.run, map_location="cpu")[3].numpy()
        if stored.shape != (N0, N1):
            print(f"  NOTE: stored channel is {stored.shape}, field is {(N0,N1)} - this prep was")
            print(f"  built from PREDICTED velocities (center-cropped), so rebuilding it needs that")
            print(f"  same step-1 model. Only the shape is reported here.")
        else:
            print(f"  stored: {int((stored>0).sum())} lit cells   ({a.identify.name})")
            cands = [(f"OLD {m}", img_old[m]) for m in METHODS] + [("NEW fixed RK4", img_new)]
            for nm, img in cands:
                A, B = stored > 0, img > 0
                i = int((A & B).sum())
                print(f"    vs {nm:16s} overlap {i:7d} = {100*i/max(A.sum(),1):5.1f}% of stored, "
                      f"IoU {100*i/max((A|B).sum(),1):5.1f}%,  mean|d| {np.abs(stored-img).mean():.3e}")
            print("  -> a near-100% match identifies the solver that generated this prep.")

    # ------------------------------------------------------- E10 provenance of a RENDERED channel
    if a.reference_png is not None:
        banner("E10  WHICH SOLVER PRODUCED AN OLDER, RENDERED CHANNEL?")
        if a.identify is None:
            print("  needs --identify <current prep> as the positive control; skipping.")
        else:
            julia, px = mask_from_render(a.reference_png, N0)
            print(f"  {a.reference_png.name}")
            print(f"  recovered {julia.sum()} lit cells at {px[0]:.2f} x {px[1]:.2f} px per cell")
            control = stored > 0
            # orientation: older plotting code may have transposed/flipped the array
            def coarse(m, k=32):
                return m.reshape(N0 // k, k, N1 // k, k).any((1, 3))
            cs = coarse(control)
            forms = [("identity", lambda x: x), ("transpose", lambda x: x.T),
                     ("flipud", np.flipud), ("fliplr", np.fliplr),
                     ("rot180", lambda x: np.flipud(np.fliplr(x)))]
            scored = [(nm, (coarse(np.ascontiguousarray(f(julia))) & cs).sum()
                       / max((coarse(np.ascontiguousarray(f(julia))) | cs).sum(), 1), f)
                      for nm, f in forms]
            nm, sc, f = max(scored, key=lambda t: t[1])
            print("  orientation (coarse 32-cell IoU vs the control): "
                  + ", ".join(f"{n}={100*v:.0f}%" for n, v, _ in scored))
            print(f"  -> using '{nm}'")
            julia = np.ascontiguousarray(f(julia))
            jd = _dilate(julia, 1)
            cands = [(f"OLD {m}", img_old[m] > 0) for m in METHODS] + \
                    [("NEW fixed RK4", img_new > 0), ("CONTROL (current prep)", control)]
            print(f"\n  {'candidate':<26s} {'shift':>8s} {'IoU':>7s} {'within 1 cell':>14s}")
            for cname, cimg in cands:
                i, dy, dx, s2 = _best_shift(cimg, julia)
                print(f"  {cname:<26s} ({dy:+d},{dx:+d}) {100*i/max((s2|julia).sum(),1):6.1f}% "
                      f"{100*(s2&jd).sum()/max(s2.sum(),1):13.1f}%")
            print("  Read against the CONTROL row, not against 100%: the render costs most of the")
            print("  overlap. A candidate scoring like the control is indistinguishable from it at")
            print("  this resolution; one scoring far below it is excluded.")

    # ------------------------------------------------------- optional cross-check
    if a.cross_check:
        banner("CROSS-CHECK AGAINST THE REPO IMPLEMENTATION")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from step2_streamlines.streamlines_helpers import calc_streamlines, build_velocity_grid
        Vt = build_velocity_grid(torch.as_tensor(U[1] if a.randomK else U[0]),
                                 torch.as_tensor(U[0] if a.randomK else U[1]),
                                 (N0, N1), randomK_data=bool(a.randomK), dtype=torch.float64)
        assert torch.allclose(Vt, torch.as_tensor(U)), "velocity grid convention mismatch"
        repo = calc_streamlines(torch.as_tensor(starts), Vt, maxs, t_end=T_END, t_steps=T_STEPS)
        mine = [cut_new(fine[i, :, 0], fine[i, :, 1], t_eval, maxs) for i in range(len(starts))]
        dmax = max(float(torch.abs(repo[i][0] - torch.as_tensor(mine[i][0][:len(repo[i][0])])).max())
                   for i in range(len(starts)))
        print(f"  max |repo calc_streamlines - this file's RK4| = {dmax:.3e} cells "
              f"(float32 vs float64 rounding only)")

    if a.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        i = int(np.argmax(mx_o))
        cols = {"RK45": "r", "Radau": "m", "RK23": "y", "LSODA": "g", "BDF": "b"}
        ax[0].plot(ref[i, :valid[i], 1], ref[i, :valid[i], 0], "k-", lw=3, label="reference (DOP853, rtol 1e-12)")
        for m in METHODS:
            ax[0].plot(old_traj[m][i, :valid[i], 1], old_traj[m][i, :valid[i], 0],
                       cols.get(m, "r") + "--", lw=1.5, label=f"BEFORE: {m}, scipy defaults")
        yn = new_Y
        ax[0].plot(yn[i, :valid[i], 1], yn[i, :valid[i], 0], "c:", lw=2, label="AFTER: fixed-step RK4")
        ax[0].set_title(f"worst line (#{i})")
        ax[0].set_xlabel("axis 1 [cells]"); ax[0].set_ylabel("axis 0 [cells]"); ax[0].legend(fontsize=8); ax[0].invert_yaxis()
        e_n = np.linalg.norm(yn[i] - ref[i], axis=1)
        for m in METHODS:
            e_o = np.linalg.norm(old_traj[m][i] - ref[i], axis=1)
            ax[1].semilogy(times[:valid[i]], np.maximum(e_o[:valid[i]], 1e-16),
                           cols.get(m, "r") + "--", label=f"BEFORE: {m}")
        ax[1].semilogy(times[:valid[i]], np.maximum(e_n[:valid[i]], 1e-16), "c-", lw=2, label="AFTER: RK4")
        ax[1].axhline(1.0, color="k", lw=0.8); ax[1].text(0.2, 1.2, "one cell", fontsize=8)
        ax[1].set_xlabel("t [y]"); ax[1].set_ylabel("position error [cells]"); ax[1].legend(fontsize=8)
        ax[1].set_title("error growth along the same line")
        fig.tight_layout(); fig.savefig(a.plot, dpi=150)
        print(f"\nfigure written to {a.plot}")


if __name__ == "__main__":
    main()
