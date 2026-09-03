"""Optional restriction of every dataset to one hardcoded sub-window of the full domain.

A debugging aid. With it on, training, validation and testing all run on a 736x736-cell corner of
the 2560x2560 field instead of the whole thing - 12.1x fewer cells per forward pass, so a
full-domain end-to-end step that takes minutes takes seconds and fits on a CPU box.

Nothing on disk changes: the prepared datasets stay full-domain and the window is applied when a
datapoint is LOADED. So switching the option off returns to the full field with no rebuild, and the
normalization statistics in info.yaml stay those of the whole domain - which is what keeps a
windowed run comparable to a full one and lets pretrained baselines load unchanged.

Where the window comes from
---------------------------
It is the region requested off the inference figures (figures/inference/*_true.png). Those plot the
network OUTPUT with `imshow`, which puts array rows on the vertical axis - labelled "x [m]" there -
and columns on the horizontal axis, labelled "y [m]". The requested region is

    x =    0 ..  2000 m   (vertical, from the top of the figure)
    y = 2000 ..  4000 m   (horizontal)

i.e. 400x400 cells at 5 m/cell. That is the frame of the plotted field, not of the raw field: the
valid (no-padding) convolutions of CNN1 (kernel 5) and CNN2 (kernel 4) remove 333 cells in total,
so the end-to-end output is 2227^2 = 11135 m across - the extent of those figures - and its origin
sits at raw cell (2560 - 2227) // 2 = 166.

Supervising that 400x400 output window therefore needs a 400 + 333 = 733-cell input window: the
requested region grown by the 166-cell halo on every side. The size is then rounded UP to 736,
the next multiple of 16, because the crop is quantized by the four pooling stages - a size that is
a multiple of 16 loses 333 cells, one that is not can lose 341 - and 2560 is a multiple of 16, so
this keeps the windowed run's crop arithmetic identical to the full-domain run's.

    rows (x)  166 - 166 ..  566 + 170   ->    0 ..  736
    cols (y)  566 - 166 ..  966 + 170   ->  400 .. 1136

The halo fits exactly at the top edge, since the requested region already starts 166 raw cells in,
so nothing is clipped. The supervised output is 403x403 cells, covering x = 0..2015 m and
y = 2000..4015 m: the requested window plus three cells at the far edges.

That output window is exact only for this two-net architecture. A model with a different crop
(baseline_v, which is CNN1 alone) supervises a LARGER window, still containing the requested region.
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

    Center-offset aware, because a prepared dataset does not always store Inputs and Labels at the
    same size: a streamline prep built with `based_on_pred=True` holds Inputs already center-cropped
    to CNN1's output while its Labels stay full-size, and every consumer aligns such a pair by
    centering. Slicing both at the same array indices would take two different physical regions, so
    the window is resolved against each tensor's own origin in the raw field instead.

    A field that has lost its border that way can no longer reach the edge of the domain, and this
    window touches the top edge. Rather than refuse it, the window is shrunk symmetrically - by the
    same amount on all four sides - until it fits. That keeps its center fixed, so the Input and
    Label of one datapoint come back differing by exactly the network's crop and still centered on
    each other, which is the relationship the rest of the code expects.
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
