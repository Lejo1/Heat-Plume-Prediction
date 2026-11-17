"""Optional restriction of every dataset to one hardcoded window of the full domain (debugging aid).

With it on, training, validation and testing run on a 736x736-cell window of the 2560x2560 field.
The window is applied when a datapoint is loaded, so the prepared datasets and their normalization
statistics stay those of the full domain.

The window is the input region needed to supervise the output region x = 0..2000 m, y = 2000..4000 m
of the end-to-end model (400 x 400 cells plus the 166-cell halo of the two valid-convolution UNets,
rounded up to a multiple of 16).
"""
import torch

# raw cell indices into the prepared 2560^2 field: x0, x1, y0, y1 (axis 0 = x, axis 1 = y)
WINDOW = (0, 736, 400, 1136)
FULL_SHAPE = (2560, 2560)

_enabled = False


def enable(flag: bool = True):
    """Turn the window on or off for this process. Call once, before any dataset is constructed."""
    global _enabled
    _enabled = bool(flag)
    return _enabled


def is_enabled() -> bool:
    return _enabled


def shape() -> tuple:
    """Spatial size of a windowed field."""
    return (WINDOW[1] - WINDOW[0], WINDOW[3] - WINDOW[2])


def describe() -> str:
    x0, x1, y0, y1 = WINDOW
    if not _enabled:
        return "subdomain: off (full domain)"
    h, w = shape()
    return (f"subdomain: ON - raw cells x {x0}..{x1}, y {y0}..{y1} ({h}x{w} of "
            f"{FULL_SHAPE[0]}x{FULL_SHAPE[1]}, {FULL_SHAPE[0] * FULL_SHAPE[1] / (h * w):.1f}x fewer "
            f"cells). Metrics from this run are NOT comparable to full-domain ones.")


def crop(t: torch.Tensor) -> torch.Tensor:
    """Cut `t` (..., H, W) down to the window. Identity while the option is off.

    The window is placed relative to each tensor's own origin in the raw field (Inputs and Labels of a
    prep can differ in size by the network's crop). If it does not fit, it is shrunk by the same amount
    on all sides, so Inputs and Labels stay centered on each other.
    """
    if not _enabled:
        return t
    h, w = t.shape[-2], t.shape[-1]
    win_h, win_w = shape()
    if h <= win_h and w <= win_w:
        return t  # already windowed (e.g. re-loaded within one process)

    off_r = (FULL_SHAPE[0] - h) // 2  # where this tensor's origin sits in the raw field
    off_c = (FULL_SHAPE[1] - w) // 2
    # how far the window overhangs this field, on its worst side
    d = max(off_r - WINDOW[0], WINDOW[1] - (off_r + h),
            off_c - WINDOW[2], WINDOW[3] - (off_c + w), 0)
    if win_h - 2 * d <= 0 or win_w - 2 * d <= 0:
        raise ValueError(
            f"subdomain window {WINDOW} cannot be fitted into a {h}x{w} field sitting at raw cells "
            f"x {off_r}..{off_r + h}, y {off_c}..{off_c + w}: it would have to shrink by {d} cells "
            f"per side, leaving nothing. Move WINDOW in preprocessing/subdomain.py away from the "
            f"domain edge, or make it larger.")
    r0 = WINDOW[0] + d - off_r
    c0 = WINDOW[2] + d - off_c
    return t[..., r0:r0 + win_h - 2 * d, c0:c0 + win_w - 2 * d]
