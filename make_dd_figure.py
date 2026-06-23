#!/usr/bin/env python3
"""
Dataset-distillation schematic figure.

  [ real CIFAR-10 grid ]  --(dataset distillation)-->  [ MTT distilled grid ]
          | Train                                              | Train
          v                                                    v
       (network)  - - - - - - similar performance - - - - - (network)

Real data : CIFAR-10 (downloaded via torchvision).
Synthetic : MTT distilled images pulled from George Cazenavette's project page:
            https://georgecazenavette.github.io/mtt-distillation/images/cifar10_10/

Usage:
    pip install torch torchvision matplotlib pillow numpy
    python make_dd_figure.py
Outputs dd_figure.png and dd_figure.pdf in the current directory.
"""

import io
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Ellipse
from PIL import Image

# ----------------------------- configuration ------------------------------- #
REAL_ROWS, REAL_COLS = 3, 6          # 3 x 6 = 18 real images
SYN_ROWS,  SYN_COLS  = 2, 4          # 2 x 4 = 8  synthetic images

# Which CIFAR-10 classes / instance index to pull for the synthetic grid.
# Files are named "<class>_<idx>.png" with idx in 0..9 on the MTT page.
MTT_BASE = "https://georgecazenavette.github.io/mtt-distillation/images/cifar10_10"
SYN_PICKS = [                        # 8 distinct classes, one distilled image each
    ("airplane", 0), ("automobile", 0), ("bird", 0), ("cat", 0),
    ("deer", 0), ("dog", 0), ("frog", 0), ("horse", 0),
]

REAL_TITLE = "Original (50K samples)"
SYN_TITLE  = "Distilled (500 samples)"

NODE_COLOR = "#F2B33D"   # gold
EDGE_COLOR = "#8A86C7"   # muted purple
INK        = "black"
OUT        = "dd_figure"
CACHE      = "dd_images.npz"   # cached real/synthetic images so reruns skip download
# ---------------------------------------------------------------------------- #


def get_real_images(n):
    """Return a list of n HxWx3 uint8 arrays from CIFAR-10, spread across classes."""
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root="./data", train=True, download=True)
    labels = np.array(ds.targets)
    chosen = []
    # round-robin across classes so the grid looks varied, like the reference figure
    by_class = {c: list(np.where(labels == c)[0]) for c in range(10)}
    k = 0
    while len(chosen) < n:
        c = k % 10
        if by_class[c]:
            chosen.append(by_class[c].pop(0))
        k += 1
    return [np.asarray(ds[i][0]) for i in chosen]


def get_synthetic_images(picks):
    """Download MTT distilled PNGs from the project page."""
    out = []
    for cls, idx in picks:
        url = f"{MTT_BASE}/{cls}_{idx}.png"
        data = urllib.request.urlopen(url, timeout=30).read()
        out.append(np.asarray(Image.open(io.BytesIO(data)).convert("RGB")))
    return out


# --------------------------- drawing primitives ----------------------------- #
def draw_grid(ax, images, x0, y0, cell, ncols, nrows, gap=0.06):
    """Place images as a tight grid; top-left cell starts at (x0, y0_top)."""
    w = ncols * cell + (ncols - 1) * gap
    h = nrows * cell + (nrows - 1) * gap
    for k, img in enumerate(images[: ncols * nrows]):
        r, c = divmod(k, ncols)
        cx = x0 + c * (cell + gap)
        cy = y0 - r * (cell + gap)          # y grows upward; rows go downward
        ax.imshow(img, extent=[cx, cx + cell, cy - cell, cy],
                  zorder=3, interpolation="nearest")
    # thin frame tightly bounding the whole block (block spans y in [y0-h, y0])
    pad = gap
    ax.add_patch(plt.Rectangle((x0 - pad, y0 - h - pad),
                               w + 2 * pad, h + 2 * pad, fill=False,
                               ec="0.45", lw=1.0, zorder=4))
    return w, h


def block_arrow(ax, x, y, dx, dy, lw_scale=0.8, head_scale=0.65):
    ax.add_patch(FancyArrow(x, y, dx, dy, width=0.10 * lw_scale,
                            head_width=0.42 * lw_scale * head_scale,
                            head_length=0.42 * lw_scale * head_scale,
                            length_includes_head=True, color=INK, zorder=5))


def draw_network(ax, cx, cy, scale=1.0):
    """Small 2-3-2 MLP doodle with gold nodes and purple edges."""
    layers = [[-0.7, 0.7], [-1.0, 0.0, 1.0], [-0.7, 0.7]]
    xs = [-1.15, 0.0, 1.15]
    pos = [[(cx + xs[li] * scale, cy + y * scale) for y in layer]
           for li, layer in enumerate(layers)]
    # edges between consecutive layers
    for li in range(len(pos) - 1):
        for a in pos[li]:
            for b in pos[li + 1]:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=EDGE_COLOR,
                        lw=2.4 * scale, solid_capstyle="round", zorder=2)
    # nodes
    for layer in pos:
        for (x, y) in layer:
            ax.add_patch(Ellipse((x, y), 0.45 * scale, 0.45 * scale,
                                 facecolor=NODE_COLOR, edgecolor="none", zorder=3))


# ------------------------------- assembly ----------------------------------- #
def build_figure(real_imgs, syn_imgs, out=OUT,
                 real_title=REAL_TITLE, syn_title=SYN_TITLE):
    fig, ax = plt.subplots(figsize=(11, 6.0))
    ax.set_xlim(0.3, 14.0)
    ax.set_ylim(3.75, 11.5)
    ax.set_aspect("equal")
    ax.axis("off")

    cell = 1.0

    # --- real grid (top-left) ---
    rx0, ry0 = 0.4, 10.7
    rw, rh = draw_grid(ax, real_imgs, rx0, ry0, cell, REAL_COLS, REAL_ROWS)
    real_cx = rx0 + rw / 2
    ax.text(real_cx, 11.25, real_title, ha="center", va="center",
            fontsize=15, fontweight="bold")

    # --- distillation arrow ---
    arrow_y = 9.2
    block_arrow(ax, rx0 + rw + 0.45, arrow_y, 2.10, 0.0, lw_scale=1.1)
    # ax.text(rx0 + rw + 1.5, arrow_y + 0.3, "Distillation",
    #         ha="center", va="center", fontsize=14, fontweight="bold")

    # --- synthetic grid (top-right) ---
    sx0 = rx0 + rw + 3.0
    sy0 = 10.25
    sw, sh = draw_grid(ax, syn_imgs, sx0, sy0, cell, SYN_COLS, SYN_ROWS)
    syn_cx = sx0 + sw / 2
    ax.text(syn_cx, sy0 + 0.7, syn_title, ha="center", va="center",
            fontsize=15, fontweight="bold")

    # --- "Train" arrows downward ---
    net_cy = 4.9
    net_top = 6.3
    block_arrow(ax, real_cx, ry0 - rh - 0.10, 0.0, -(ry0 - rh - net_top), lw_scale=1.1)
    ax.text(real_cx + 0.35, (ry0 - rh + net_top) / 2 + 0.051, "Train",
            ha="center", va="center", rotation=-90, fontsize=15, fontweight="bold")

    block_arrow(ax, syn_cx, sy0 - sh - 0.10, 0.0, -(sy0 - sh - net_top), lw_scale=1.1)
    ax.text(syn_cx + 0.35, (sy0 - sh + net_top) / 2 + 0.1, "Train",
            ha="center", va="center", rotation=-90, fontsize=15, fontweight="bold")

    # --- networks ---
    draw_network(ax, real_cx, net_cy, scale=0.85)
    draw_network(ax, syn_cx, net_cy, scale=0.85)

    # --- "Similar performance" dotted link ---
    ax.plot([real_cx + 1.7, syn_cx - 1.7], [net_cy, net_cy],
            ls=(0, (1, 2)), color=INK, lw=2.0, zorder=2)
    ax.text((real_cx + syn_cx) / 2, net_cy + 0.5, "Similar performance on test set",
            ha="center", va="center", fontsize=15, fontweight="bold")

    fig.tight_layout(pad=0)
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", pad_inches=0.0)
    fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.0)
    print(f"saved {out}.png and {out}.pdf")
    return fig


def load_images(cache=CACHE):
    """Fetch real + synthetic images once and cache them.

    First run downloads CIFAR-10 and the MTT PNGs (slow) and writes `cache`.
    Later runs load straight from `cache` (fast) so you can iterate on the
    figure styling. Delete the cache file to force a refresh.
    """
    import os
    if os.path.exists(cache):
        data = np.load(cache, allow_pickle=True)
        real = [np.asarray(x, dtype=np.uint8) for x in data["real"]]
        syn = [np.asarray(x, dtype=np.uint8) for x in data["syn"]]
        return real, syn
    real_imgs = get_real_images(REAL_ROWS * REAL_COLS)
    syn_imgs = get_synthetic_images(SYN_PICKS)
    np.savez(cache,
             real=np.stack(real_imgs).astype(np.uint8),
             syn=np.stack(syn_imgs).astype(np.uint8))
    print(f"cached images to {cache}")
    return real_imgs, syn_imgs


def main():
    real_imgs, syn_imgs = load_images()
    build_figure(real_imgs, syn_imgs)


if __name__ == "__main__":
    main()