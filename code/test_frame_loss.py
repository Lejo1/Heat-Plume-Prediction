"""
Tests of the streamline-frame losses (FrameLoss: L_N, L_TOF) on small synthetic fields, CPU, float64.

Checks:
1. u = v*                    => L_N = 0 and L_TOF = 0
2. u = R(alpha) v*           => L_N = sin^2(alpha) * V*^2 / (V*^2 + eps^2), L_TOF = (log cos alpha)^2
3. u = (1 + c) v*            => L_N = 0, L_TOF = log(1 + c)^2
4. gradient w.r.t. the predicted velocity: only on cells next to true-line nodes, L_N's orthogonal
   and L_TOF's parallel to t*, autograd = central finite differences
5. no gradient reaches the simulated velocity (true lines are constants)
6. well mask: no kept node within r_w of a pump; the flow-based radius grows where |v*| is small
7. seeds: hp reproduces the heat pumps, random stays in the box and is reproducible in eval mode
8. lambda_N = lambda_TOF = 0 => E2ELoss is bit-identical to the loss without frame terms
plus: uniform arc-length resampling, tangent along the line, randomK axis convention.

Usage:
    python test_frame_loss.py
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parent))
from processing.loss_fcts import (E2ELoss, FrameLoss, frame_terms, trace_true_lines, well_mask,
                                  well_parameters, SEED_BOX_RAW)
from step2_streamlines.streamlines_helpers import STREAMLINE_CELL_SIZE

torch.set_default_dtype(torch.float64)
H, W = 96, 112
DATA_RAW = Path(__file__).parents[2] / "datasets" / "dataset_giant_100hp_varyK"


def stub(hp, randomK=False, training=True):
    """The model attributes FrameLoss reads; v_delta 1 / v_min 0, so the velocity channels are m/y."""
    return SimpleNamespace(last_hp=[hp], last_hp_all=[hp], v_delta=torch.ones(2, 1, 1), v_min=torch.zeros(2, 1, 1),
                           randomK_data=randomK, t_steps=2000, use_compile=False, offsets=(0, 10, -10),
                           training=training)


def field(direction=None):
    """[1, 3, H, W] label [T, vx, vy] in m/y: a curved, non-uniform flow (or a uniform direction with
    varying speed), everywhere much faster than eps."""
    i = torch.arange(H)[:, None].expand(H, W)
    j = torch.arange(W)[None, :].expand(H, W)
    if direction is None:
        vx = 60 + 25 * torch.sin(j / 13.0)
        vy = 20 + 15 * torch.cos(i / 17.0)
    else:
        speed = 50 + 20 * torch.sin(i / 11.0) * torch.cos(j / 7.0)
        vx, vy = speed * direction[0], speed * direction[1]
    return torch.stack([torch.zeros(H, W), vx, vy])[None]


def with_v(y, vx, vy):
    return torch.cat([y[:, :1], vx[None, None], vy[None, None]], dim=1)


def make(hp, randomK=False, **options):
    opts = dict(frame_eps=0.5, seed_mode="hp", well_mask_radius=3.0, tof_checkpoint_cells=10,
                seed_box_raw=((0, H - 1), (0, W - 1)))  # the real pump box lies outside this small field
    opts.update(options)
    return FrameLoss(stub(hp, randomK), raw_origin=(0, 0), field_shape=(H, W), **opts)


HP = torch.tensor([[5.5, 8.5], [20.5, 40.5], [10.5, 70.5], [40.5, 20.5]])


def check(name, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {detail}")
    assert ok, name


def test_resampling_and_frame():
    print("resampling / frame")
    y = field()
    fl = make(HP)
    line = fl._line(0, y, True)
    nodes, valid = line["nodes"], line["valid"]
    check("lines have nodes", valid.sum() > 100, f"({int(valid.sum())} nodes on {valid.shape[0]} lines)")
    d = (nodes[:, 1:] - nodes[:, :-1]).norm(dim=-1)[valid[:, 1:]]
    check("uniform arc length", (d - 0.5).abs().max() < 0.02, f"(spacing {float(d.min()):.4f}..{float(d.max()):.4f}, ds 0.5)")
    step = nodes[:, 1:] - nodes[:, :-1]
    cos = (step / step.norm(dim=-1, keepdim=True) * line["t"][:, :-1]).sum(-1)[valid[:, 1:]]
    check("tangent t* along the line", cos.min() > 0.99, f"(min cos {float(cos.min()):.4f})")
    for randomK in (False, True):
        fl = make(HP, randomK=randomK)
        line = fl._line(0, y, True)
        v = fl._grid(y[0, 1:3])
        # randomK: tracer axis 0 carries vy; the grid and so the frame follow the tracer's convention
        expect = (y[0, 2] if randomK else y[0, 1]) / STREAMLINE_CELL_SIZE
        check(f"grid axis 0 (randomK {randomK})", torch.equal(v[0], expect))


def test_identity_rotation_scaling():
    print("tests 1-3")
    for randomK in (False, True):
        y = field()
        fl = make(HP, randomK=randomK)
        r = fl(y.clone(), y)
        check(f"1. u = v*: L_N = 0, L_TOF = 0 (randomK {randomK})",
              float(r["L_N"]) < 1e-24 and float(r["L_TOF"]) < 1e-24,
              f"(L_N {float(r['L_N']):.1e}, L_TOF {float(r['L_TOF']):.1e})")

    y = field()
    fl = make(HP)
    line = fl._line(0, y, True)
    eps = fl.shared["eps"]
    m, V = line["m"], line["V"]
    for alpha_deg in (5.0, 20.0, 45.0):
        a = math.radians(alpha_deg)
        vx, vy = y[0, 1], y[0, 2]
        u = with_v(y, math.cos(a) * vx - math.sin(a) * vy, math.sin(a) * vx + math.cos(a) * vy)
        r = fl(u, y)
        expect_N = float((m * math.sin(a) ** 2 * V ** 2 / (V ** 2 + eps ** 2)).sum() / m.sum())
        expect_TOF = math.log(math.cos(a)) ** 2
        check(f"2. rotation {alpha_deg:4.1f} deg", abs(float(r["L_N"]) - expect_N) < 1e-10 * expect_N
              and abs(float(r["L_TOF"]) - expect_TOF) < 1e-8 * expect_TOF and abs(float(r["angle [deg]"]) - alpha_deg) < 1e-6,
              f"(L_N {float(r['L_N']):.6e} vs {expect_N:.6e}, L_TOF {float(r['L_TOF']):.6e} vs {expect_TOF:.6e}, "
              f"angle {float(r['angle [deg]']):.4f})")
    for c in (-0.3, 0.1, 0.5):
        u = with_v(y, (1 + c) * y[0, 1], (1 + c) * y[0, 2])
        r = fl(u, y)
        expect = math.log(1 + c) ** 2
        check(f"3. scaling 1{c:+.1f}", float(r["L_N"]) < 1e-24 and abs(float(r["L_TOF"]) - expect) < 1e-8 * expect,
              f"(L_N {float(r['L_N']):.1e}, L_TOF {float(r['L_TOF']):.6e} vs {expect:.6e})")


def test_gradients():
    print("test 4/5: gradients")
    direction = torch.tensor([0.8, 0.6])
    y = field(direction)
    torch.manual_seed(0)
    u0 = y.clone()
    u0[:, 1:3] += 8 * torch.randn(1, 2, H, W)
    fl = make(HP)
    line = fl._line(0, y, True)
    # cells next to kept nodes, with the index arithmetic of sample_velocity
    nodes = line["nodes"][line["m"]]
    i = nodes[:, 0].clamp(0, H - 1).floor().long().clamp(max=H - 2)
    j = nodes[:, 1].clamp(0, W - 1).floor().long().clamp(max=W - 2)
    near = torch.zeros(H, W, dtype=torch.bool)
    for di in (0, 1):
        for dj in (0, 1):
            near[i + di, j + dj] = True
    grads = {}
    for name in ("L_N", "L_TOF"):
        u = u0.clone().requires_grad_(True)
        y_leaf = y.clone().requires_grad_(True)
        fl(u, y_leaf)[name].backward()
        g = u.grad[0, 1:3]
        grads[name] = g
        check(f"4. {name}: gradient only next to true-line nodes",
              g[:, ~near].abs().max() == 0 and g[:, near].abs().max() > 0,
              f"({int((g.abs().sum(0) > 0).sum())} cells with gradient, {int(near.sum())} adjacent cells)")
        check(f"5. {name}: no gradient into the true lines", y_leaf.grad is None)
    gN, gT = grads["L_N"], grads["L_TOF"]
    # randomK False: label channels (vx, vy) = tracer axes (0, 1), so t* = direction everywhere
    par_N = (gN * direction[:, None, None]).sum(0).abs().max() / gN.norm(dim=0).max()
    cross_T = (gT[0] * direction[1] - gT[1] * direction[0]).abs().max() / gT.norm(dim=0).max()
    check("4. L_N gradient orthogonal to t*", par_N < 1e-12, f"(max |g.t|/|g| {float(par_N):.1e})")
    check("4. L_TOF gradient parallel to t*", cross_T < 1e-12, f"(max |g x t|/|g| {float(cross_T):.1e})")

    # autograd vs central finite differences on a few cells with gradient
    for name in ("L_N", "L_TOF"):
        g = grads[name]
        cells = torch.nonzero(g.abs().sum(0) > 0)
        pick = cells[torch.randperm(len(cells), generator=torch.Generator().manual_seed(1))[:8]]
        worst = 0.0
        for ci, cj in pick.tolist():
            for ch in (0, 1):
                h = 1e-5
                up, dn = u0.clone(), u0.clone()
                up[0, 1 + ch, ci, cj] += h
                dn[0, 1 + ch, ci, cj] -= h
                fd = (float(fl(up, y)[name]) - float(fl(dn, y)[name])) / (2 * h)
                ad = float(g[ch, ci, cj])
                worst = max(worst, abs(fd - ad) / max(abs(ad), 1e-12))
        check(f"4. {name}: autograd = finite differences", worst < 1e-5, f"(max rel. error {worst:.1e})")


def test_mask():
    print("test 6: well mask")
    y = field()
    fl = make(HP, well_mask_radius=4.0)
    line = fl._line(0, y, True)
    kept = line["nodes"][line["m"]]
    dmin = torch.cdist(kept, HP).min(dim=1).values
    dropped = (line["valid"] & ~line["m"]).sum()
    check("6. fixed: no kept node within r_w of a pump", dmin.min() >= 4.0,
          f"(closest kept node {float(dmin.min()):.2f} cells, {int(dropped)} nodes masked)")

    params = {"Q": 0.00024 * 365.25 * 24 * 3600, "b": 5.0, "phi": 0.25}
    pump = torch.tensor([[50.0, 50.0]])
    ring = torch.tensor([[[50.0 + r, 50.0] for r in (1.5, 3.0, 6.0, 10.0, 20.0)]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    for v_m, expect in ((1000.0, [False, True, True, True, True]), (100.0, [False, False, False, False, True]),
                        (30.0, [False, False, False, False, False])):
        V = torch.full((1, 5), v_m / STREAMLINE_CELL_SIZE)
        m = well_mask(ring, valid, pump, V, mode="flow", c=1.0, params=params)
        r_w = max(2.0, params["Q"] / (params["b"] * params["phi"] * v_m) / STREAMLINE_CELL_SIZE)
        check(f"6. flow: |v*| = {v_m:6.1f} m/y -> r_w {r_w:5.1f} cells", m[0].tolist() == expect,
              f"(kept at distances 1.5/3/6/10/20: {m[0].tolist()})")
    if (DATA_RAW / "inputs" / "settings.yaml").exists():
        wp = well_parameters(DATA_RAW)
        check("6. well_parameters from the simulation", abs(wp["Q"] - params["Q"]) < 1 and wp["b"] == 5.0
              and wp["phi"] == 0.25, f"({wp})")


def test_seeds():
    print("test 7: seeds")
    fl = make(HP)
    s = fl.seeds(0, (H, W), training=True)
    check("7. hp = heat-pump positions", torch.equal(s, HP))
    s = make(HP, seed_hp_offsets=True).seeds(0, (H, W), training=True)
    inside = lambda p: (p[:, 0] >= 0) & (p[:, 0] <= H - 1) & (p[:, 1] >= 0) & (p[:, 1] <= W - 1)
    expect = torch.cat([HP + torch.tensor([0.0, o]) for o in (0, 10, -10)])
    check("7. hp + offsets", torch.equal(s, expect[inside(expect)]), f"({len(s)} seeds)")

    # random: raw box placed into a 2227^2 output region at raw origin 166 (full-domain e2e)
    region, field_shape = (2227, 2227), (2560, 2560)
    fr = FrameLoss(stub(HP), raw_origin=(0, 0), field_shape=field_shape, seed_mode="random", n_seeds=500, seed_rng=3)
    a = fr.seeds(0, region, training=False)
    b = fr.seeds(0, region, training=False)
    raw = a + 166
    in_box = all(((raw[:, ax] >= SEED_BOX_RAW[ax][0]) & (raw[:, ax] <= SEED_BOX_RAW[ax][1] + 1)).all() for ax in range(2))
    in_region = ((a >= 0) & (a <= region[0] - 1)).all()
    check("7. random seeds inside the pump box and the output region", in_box and in_region,
          f"(raw axis0 {float(raw[:, 0].min()):.1f}..{float(raw[:, 0].max()):.1f}, "
          f"axis1 {float(raw[:, 1].min()):.1f}..{float(raw[:, 1].max()):.1f})")
    check("7. random seeds are cell centres", torch.equal(a - a.floor(), torch.full_like(a, 0.5)))
    check("7. eval seeds identical across calls", torch.equal(a, b))
    c = fr.seeds(0, region, training=True)
    d = fr.seeds(0, region, training=True)
    check("7. training seeds redrawn", not torch.equal(c, d))
    fixed = FrameLoss(stub(HP), raw_origin=(0, 0), field_shape=field_shape, seed_mode="random", seed_rng=3,
                      resample_seeds_each_epoch=False)
    check("7. resample off: training seeds fixed",
          torch.equal(fixed.seeds(0, region, training=True), fixed.seeds(0, region, training=True)))
    both = make(HP, seed_mode="both", n_seeds=7).seeds(0, (H, W), training=False)
    check("7. both = hp + random", len(both) == len(HP) + 7 and torch.equal(both[:len(HP)], HP))

    # training cache: random lines are re-traced when the epoch changes, hp lines are not
    y = field()
    fr = make(HP, seed_mode="random", n_seeds=20)
    l0 = fr._line(0, y, True)
    check("7. cache hit within an epoch", fr._line(0, y, True) is l0)
    fr.set_epoch(1)
    check("7. new random lines in a new epoch", fr._line(0, y, True) is not l0)


def test_defaults_bit_identical():
    print("test 8: defaults")
    torch.manual_seed(2)
    pred, label = torch.rand(1, 3, 40, 40, dtype=torch.float32), torch.rand(1, 3, 40, 40, dtype=torch.float32)
    for base in ("mse", "mae", "huber"):
        old = E2ELoss(lambda_v=0.3, base=base)
        new = E2ELoss(lambda_v=0.3, base=base, lambda_N=0.0, lambda_TOF=0.0, frame=make(HP))
        manual = old.fn(pred[:, 0:1], label[:, 0:1]) + 0.3 * old.fn_v(pred[:, 1:3], label[:, 1:3])
        check(f"8. {base}: lambdas 0 -> bit-identical", torch.equal(old(pred, label), new(pred, label))
              and torch.equal(old(pred, label), manual))


if __name__ == "__main__":
    test_resampling_and_frame()
    test_identity_rotation_scaling()
    test_gradients()
    test_mask()
    test_seeds()
    test_defaults_bit_identical()
    print("ALL CHECKS PASSED")
