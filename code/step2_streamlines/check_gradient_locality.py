#!/usr/bin/env python3
"""Where does dL/dv come from? Is the streamline gradient LOCAL or CHAINED along the trajectory?

The RK4 recurrence is   x_{n+1} = x_n + dt/6 (k1 + 2k2 + 2k3 + k4),  k1 = v(x_n), ...
so x_{n+1} depends on x_n twice:

    directly, through the carried "+ x_n"          -> identity
    indirectly, through v(x_n)                     -> dt * dv/dx

Backprop therefore multiplies  dx_{n+1}/dx_n = I + dt*J(x_n)  all the way back to the seed, i.e.
it solves the ODE's adjoint (variational) equation. A perturbation of the very last sample is
propagated back over the whole line, and the product of Jacobians can grow exponentially in a
shearing flow - the usual exploding/vanishing-gradient story of an RNN unrolled n_int times.

The alternative ("local" / truncated) gradient detaches the carried position, so each step blames
only the velocity it sampled itself:

    x_{n+1} = x_n.detach() + dt/6 (k1 + 2k2 + 2k3 + k4)   with every stage position detached

This file measures which of the two the repository actually implements, and what the difference is.
It is a diagnostic, not part of training.

Usage
    python check_gradient_locality.py              # all checks on a small synthetic field
    python check_gradient_locality.py --plot out.png
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step2_streamlines.streamlines_helpers import (calc_streamlines, sample_velocity,
                                                   draw_streamlines_soft)

N = 64          # grid cells per axis
T_END = 4.0
T_STEPS = 400


# ------------------------------------------------------------------------------------------------
def make_field(n=N, seed=0):
    """A smooth, shearing velocity field: long trajectories that cross many cells."""
    ii, jj = torch.meshgrid(torch.arange(n, dtype=torch.float64),
                            torch.arange(n, dtype=torch.float64), indexing="ij")
    u0 = 3.0 + 1.5 * torch.sin(2 * np.pi * jj / n)              # along axis 0, sheared by axis 1
    u1 = 1.2 * torch.cos(2 * np.pi * ii / n) + 0.4              # wiggle across
    return torch.stack([u0, u1])


def rk4_local(x, velocity, dt):
    """The 'local' step: identical forward values, but every position used for sampling - and the
    carried position - is detached, so no gradient crosses a step boundary."""
    xd = x.detach()
    k1 = sample_velocity(velocity, xd)
    k2 = sample_velocity(velocity, (xd + dt / 2 * k1).detach())
    k3 = sample_velocity(velocity, (xd + dt / 2 * k2).detach())
    k4 = sample_velocity(velocity, (xd + dt * k3).detach())
    return xd + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def trace(velocity, start, n_int, dt, local=False):
    """Integrate one seed, returning the full trajectory as a stacked tensor (n_int+1, 2)."""
    x = start.clone()
    out = [x]
    for _ in range(n_int):
        if local:
            x = rk4_local(x, velocity, dt)
        else:
            k1 = sample_velocity(velocity, x)
            k2 = sample_velocity(velocity, x + dt / 2 * k1)
            k3 = sample_velocity(velocity, x + dt / 2 * k2)
            k4 = sample_velocity(velocity, x + dt * k3)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(x)
    return torch.cat(out, dim=0)


def grad_support(velocity, traj_nodes, loss):
    """dL/dv for one loss, plus where in the grid it is non-zero."""
    if velocity.grad is not None:
        velocity.grad = None
    loss.backward(retain_graph=True)
    g = velocity.grad.abs().sum(0)                     # (N, N), summed over the two components
    nz = torch.nonzero(g > 0).double()
    return g, nz


def graph_stats(t, max_nodes=400_000):
    """Walk the autograd graph backwards from `t`.

    Returns (node count, LONGEST path, op histogram). The longest path is the one that matters:
    it is the number of backward ops that must run in sequence, i.e. how deep the recurrence is.
    (A breadth-first distance would report the SHORTEST route to each node and badly understate it.)
    Computed by an iterative DFS with memoisation, so no recursion limit and no revisits.
    """
    if t.grad_fn is None:
        return 0, 0, {}
    hist, longest, seen = {}, {}, set()
    stack = [(t.grad_fn, False)]
    while stack:
        fn, done = stack.pop()
        if done:
            best = 0
            for nxt, _ in fn.next_functions:
                if nxt is not None:
                    best = max(best, longest.get(nxt, 0) + 1)
            longest[fn] = best
            continue
        if fn in seen:
            continue
        seen.add(fn)
        hist[type(fn).__name__] = hist.get(type(fn).__name__, 0) + 1
        stack.append((fn, True))
        if len(seen) < max_nodes:
            for nxt, _ in fn.next_functions:
                if nxt is not None and nxt not in seen:
                    stack.append((nxt, False))
    return len(seen), longest.get(t.grad_fn, 0), hist


def banner(s):
    print("\n" + "=" * 92 + f"\n{s}\n" + "=" * 92)


# ------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-int", type=int, default=120, help="RK4 steps (kept small so the graph walk is cheap)")
    ap.add_argument("--real", action="store_true",
                    help="use the real 2560^2 velocity field and a real heat-pump seed instead of "
                         "the synthetic one - this is where the exploding-gradient question is "
                         "actually decided")
    ap.add_argument("--prep", type=Path,
                    default=Path("../../datasets_prep/dataset_giant_100hp_varyK inputs_pki outputs_xy"))
    ap.add_argument("--crop", type=int, default=768,
                    help="with --real: window of the 2560^2 field kept around the seed. Each "
                         "backward through sample_velocity allocates a zero buffer the size of the "
                         "velocity grid, so a full-domain CPU run is memory-bandwidth bound; the "
                         "window keeps the physics and makes it fast.")
    ap.add_argument("--t-end", type=float, default=None, help="override the integration time")
    ap.add_argument("--plot", type=Path, default=None)
    a = ap.parse_args()
    torch.manual_seed(0)

    global N, T_END
    if a.real:
        from analyze_rk_solver import load_field, T_END as REAL_T_END
        Un, starts_all = load_field(a.prep, "RUN_1.pt", True)
        U = torch.as_tensor(Un, dtype=torch.float64)
        seed = starts_all[0]
        c = a.crop
        r0 = int(np.clip(seed[0] - c // 2, 0, U.shape[1] - c))
        c0 = int(np.clip(seed[1] - c // 2, 0, U.shape[2] - c))
        U = U[:, r0:r0 + c, c0:c0 + c].contiguous()
        start = torch.tensor([[seed[0] - r0, seed[1] - c0]], dtype=torch.float64)
        N = U.shape[1]
        T_END = a.t_end if a.t_end is not None else 5.0
        field_desc = (f"REAL field {a.prep.name}/RUN_1, real heat-pump seed, "
                      f"{c}x{c} window at raw ({r0},{c0})")
    else:
        U = make_field()
        start = torch.tensor([[8.0, 8.0]], dtype=torch.float64)
        field_desc = "synthetic smooth shear field"
    dt = T_END / a.n_int

    banner("SETUP")
    print(f"  {field_desc}")
    print(f"  grid {U.shape[1]}x{U.shape[2]}, {a.n_int} RK4 steps of dt={dt:.5f} over t_end={T_END}")
    print(f"  seed at {[round(z,2) for z in start.tolist()[0]]}")
    vmax = float(U.norm(dim=0).max())
    print(f"  |v|max {vmax:.2f} cells/y  ->  {dt*vmax:.3f} cells per step "
          f"(production targets {0.5})")

    # ------------------------------------------------------------------ 1. what the repo does
    banner("1  WHAT THE REPOSITORY ACTUALLY COMPUTES")
    print("  Loss = the FINAL position only. If the gradient were local, dL/dv could only be")
    print("  non-zero in the handful of cells the LAST step sampled. Anything further back is")
    print("  proof that the gradient chains through the trajectory.")
    for label, local in (("repo (full backprop)", False), ("'local' variant (detached)", True)):
        v = U.clone().requires_grad_(True)
        traj = trace(v, start, a.n_int, dt, local=local)
        loss = traj[-1].sum()
        g, nz = grad_support(v, traj, loss)
        end = traj[-1].detach()
        d = (nz - end).norm(dim=1) if len(nz) else torch.zeros(1)
        path_len = float((traj[1:] - traj[:-1]).detach().norm(dim=1).sum())
        print(f"\n  {label}")
        print(f"     non-zero dL/dv cells : {len(nz)}")
        print(f"     farthest from the endpoint : {float(d.max()):.1f} cells "
              f"(trajectory is {path_len:.1f} cells long)")
        print(f"     -> {'CHAINED over the whole line' if float(d.max()) > 5 else 'LOCAL to the last step'}")

    # ------------------------------------------------------------------ 1b. the repo FLAG
    banner("1b  THE detach_trajectory FLAG, THROUGH THE REAL calc_streamlines")
    print("  Same test, but driving the shipped calc_streamlines instead of this file's own loop.")
    maxs = (N - 1.0, N - 1.0)
    res = {}
    for flag in (False, True):
        v = U.clone().requires_grad_(True)
        # pin n_int to a.n_int: calc_streamlines derives it as ceil(t_end*v_max/max_step_cells)
        msc = T_END * float(U.norm(dim=0).max()) / a.n_int
        lines = calc_streamlines(start, v, maxs, t_end=T_END, t_steps=a.n_int + 2,
                                 max_step_cells=msc, detach_trajectory=flag)
        sx, sy, _ = lines[0]
        if v.grad is not None:
            v.grad = None
        (sx[-1] + sy[-1]).backward()
        g = v.grad.abs().sum(0)
        nz = torch.nonzero(g > 0).double()
        endp = torch.stack([sx[-1], sy[-1]]).detach()
        far = float((nz - endp).norm(dim=1).max()) if len(nz) else 0.0
        res[flag] = (torch.stack([sx, sy], 1).detach().clone(), g.detach().clone(), len(nz), far)
        print(f"    detach_trajectory={str(flag):5s}  non-zero dL/dv cells {len(nz):5d}   "
              f"farthest from endpoint {far:7.1f} cells")

    same = torch.equal(res[False][0], res[True][0])
    print(f"\n  forward positions bit-identical between the two modes: {same}   "
          f"{'OK' if same else 'MISMATCH - detaching must not change the physics!'}")
    assert same, "detach_trajectory changed the forward pass"
    assert res[True][3] < 5 < res[False][3], "the flag did not localise the gradient"
    r = float(res[False][1].sum() / res[True][1].sum())
    print(f"  |dL/dv| total, full / truncated = {r:.1f}x   (measured for THIS variant:")
    print("     cross-step cut only, the within-step RK4 stage terms are kept)")

    # ------------------------------------------------------------------ 2. the graph itself
    banner("2  THE AUTOGRAD GRAPH")
    v = U.clone().requires_grad_(True)
    traj = trace(v, start, a.n_int, dt)
    n_nodes, depth, hist = graph_stats(traj[-1].sum())
    print(f"  full backprop : {n_nodes} nodes, longest path {depth}  ({depth/a.n_int:.0f} sequential ops per RK4 step)")
    v2 = U.clone().requires_grad_(True)
    traj2 = trace(v2, start, a.n_int, dt, local=True)
    n2, d2, _ = graph_stats(traj2[-1].sum())
    print(f"  'local'       : {n2} nodes, longest path {d2}  ({d2/a.n_int:.1f} per step - the chain is cut)")
    print(f"  most frequent ops: " + ", ".join(f"{k}x{v}" for k, v in
                                               sorted(hist.items(), key=lambda t: -t[1])[:6]))
    print(f"\n  In production n_int = 8558, so the real chain is ~{8558/a.n_int:.0f}x longer than this")
    print(f"  demo: roughly {int(depth*8558/a.n_int):,} backward ops that must run IN SEQUENCE.")
    print("  To inspect it yourself:  print(tensor.grad_fn), then walk .next_functions (see")
    print("  graph_stats() above), or `pip install torchviz` and use make_dot(tensor).")

    # ------------------------------------------------------------------ 3. amplification
    banner("3  DOES THE CHAIN AMPLIFY?  (dL/d x_n for a loss on the final position)")
    v = U.clone().requires_grad_(True)
    x = start.clone()
    nodes = [x]
    for _ in range(a.n_int):
        k1 = sample_velocity(v, x); k2 = sample_velocity(v, x + dt / 2 * k1)
        k3 = sample_velocity(v, x + dt / 2 * k2); k4 = sample_velocity(v, x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x.retain_grad()
        nodes.append(x)
    nodes[-1].sum().backward()
    mags = np.array([float(nd.grad.norm()) for nd in nodes[1:]])
    print(f"  |dL/dx_n| from the seed end to the final step:")
    for k in [0, len(mags)//8, len(mags)//4, len(mags)//2, 3*len(mags)//4, len(mags)-1]:
        print(f"     step {k:4d}/{len(mags)}  {mags[k]:.4f}")
    a_real = a.real
    amp = mags[0] / mags[-1]
    print(f"  ratio |dL/dx_first| / |dL/dx_last| = {amp:.4g}")
    if amp > 2:
        print("  >1: a perturbation at the seed matters MORE than one at the tip - the adjoint")
        print("  AMPLIFIES backwards. This is the exploding-gradient mechanism the note describes.")
    elif amp < 0.5:
        print("  <1: the adjoint DECAYS backwards; early velocities barely affect the tip here.")
    else:
        print(f"  ~1: over these {len(mags)} steps the adjoint grows only {amp:.2f}x backwards - the")
        print("  chain is long but WELL CONDITIONED here. Amplification is a property of the flow")
        print("  (shear/chaos), not of the chain as such: a physically smooth velocity field gives")
        print("  a tame adjoint, a randomly-initialised CNN1 does not." +
              ("" if a_real else " Re-check with --real."))

    # ------------------------------------------------------------------ 4. realistic loss
    banner("4  WITH THE REAL LOSS (the drawn soft-occupancy image, as in training)")
    print("  Here every sample position enters the loss, so BOTH variants put gradient along the")
    print("  whole line. The difference is what each cell's value MEANS:")
    print("    full  : v(x_n) is blamed for this sample AND for every sample downstream of it")
    print("    local : v(x_n) is blamed only for the sample it produced")
    print("  Measured through the shipped calc_streamlines + detach_trajectory, i.e. exactly what")
    print("  training runs - so this ratio is the one that matters for choosing lr_stage2.")
    res = {}
    msc = T_END * float(U.norm(dim=0).max()) / a.n_int
    for label, flag in (("full", False), ("local", True)):
        v = U.clone().requires_grad_(True)
        lines = calc_streamlines(start, v, maxs, t_end=T_END, t_steps=a.n_int + 2,
                                 max_step_cells=msc, detach_trajectory=flag)
        sx, sy, tt = lines[0]
        occ = draw_streamlines_soft([(sx, sy, tt)], (N, N), faded=True,
                                    sigma=1.0, fade_mode="absolute", t_end=T_END)
        if v.grad is not None:
            v.grad = None
        occ.sum().backward()
        g = v.grad.abs().sum(0)
        res[label] = g.detach().clone()
        print(f"\n  {label:5s}: |dL/dv| total {float(g.sum()):.4e}, "
              f"max {float(g.max()):.4e}, non-zero cells {int((g>0).sum())}")
    ratio = float(res["full"].sum() / res["local"].sum())
    print(f"\n  total |dL/dv|  full / local = {ratio:.2f}x")
    corr = float(torch.corrcoef(torch.stack([res["full"].flatten(), res["local"].flatten()]))[0, 1])
    print(f"  correlation of the two gradient maps: {corr:.4f}")
    print("  The local gradient is a DIFFERENT (truncated) descent direction, not a rescaling of")
    print("  the true one - the same trade-off as truncated BPTT in an RNN.")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        v = U.clone().requires_grad_(True)
        traj = trace(v, start, a.n_int, dt)
        g_end, _ = grad_support(v, traj, traj[-1].sum())
        v2 = U.clone().requires_grad_(True)
        traj2 = trace(v2, start, a.n_int, dt, local=True)
        g_end_l, _ = grad_support(v2, traj2, traj2[-1].sum())
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        tr = traj.detach().numpy()
        for k, (g, ttl) in enumerate([(g_end, "repo: full backprop"), (g_end_l, "'local' variant")]):
            im = ax[k].imshow(np.log10(g.detach().numpy() + 1e-20), cmap="magma")
            ax[k].plot(tr[:, 1], tr[:, 0], "c-", lw=1)
            ax[k].plot(tr[-1, 1], tr[-1, 0], "wo", ms=6)
            ax[k].set_title(f"{ttl}\nlog10 |dL/dv|, loss = final position")
            plt.colorbar(im, ax=ax[k], fraction=0.046)
        ax[2].semilogy(mags)
        ax[2].set_xlabel("RK4 step n"); ax[2].set_ylabel("|dL/dx_n|")
        ax[2].set_title("adjoint magnitude along the line\n(loss on the final position)")
        fig.tight_layout(); fig.savefig(a.plot, dpi=150)
        print(f"\nfigure written to {a.plot}")


if __name__ == "__main__":
    main()
