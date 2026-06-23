#!/usr/bin/env python3
"""
Poster-ready "Motivation / Goal" figures for the COBRA paper.
All numbers are taken from Table 1 / Table 2 of the paper (EOD, lower is better).

Run:  python motivation_figs.py
Produces fig1..fig5 as .pdf and .png. Each plot lives in its own function so you
can copy just the one you like.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ----------------------------- shared style -------------------------------- #
mpl.rcParams.update({
    "font.size": 16,
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "xtick.major.width": 1.3, "ytick.major.width": 1.3,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})
C_VAN  = "#E1812C"   # Vanilla DD (orange)
C_FAIR = "#5B7DB1"   # FairDD (steel blue)
C_COB  = "#2E8B57"   # COBRA (green)
C_FULL = "#3b3b3b"   # Full data (dark grey)
C_ACC  = "#4F2683"   # Western purple accent


def _save(fig, name):
    fig.savefig(f"{name}.pdf")
    fig.savefig(f"{name}.png", dpi=300)
    plt.close(fig)
    print("saved", name)


def _bar_labels(ax, bars, fmt="{:.0f}", dy=1.0, fs=12):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fs, fontweight="bold")


# ====================== FIG 1 : amplification (Full vs Vanilla vs COBRA) ===== #
def fig1_amplification():
    # DM backbone, IPC = 100, EOD (lower is better)
    datasets = ["CIFAR10-S", "C-FMNIST\n(FG)", "C-MNIST\n(BG)", "BFFHQ"]
    full = [48.96, 78.40, 10.30, 64.00]
    van  = [82.87, 100.0, 100.0, 63.47]
    cob  = [9.37, 24.17, 7.58, 7.87]

    x = np.arange(len(datasets)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    b1 = ax.bar(x - w, full, w, label="Full data", color=C_FULL)
    b2 = ax.bar(x,     van,  w, label="Vanilla DD (DM)", color=C_VAN)
    b3 = ax.bar(x + w, cob,  w, label="DM + COBRA", color=C_COB)
    for bars in (b1, b2, b3):
        _bar_labels(ax, bars)
    ax.set_ylabel("EOD  (lower is better) $\\downarrow$")
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylim(0, 112)
    ax.set_title("Distillation amplifies bias — COBRA reverses it",
                 fontsize=17, fontweight="bold", pad=12)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=13)
    _save(fig, "fig1_amplification")


# ====================== FIG 2 : method comparison (Vanilla/FairDD/COBRA) ===== #
def fig2_methods():
    # DM backbone, IPC = 50, EOD
    datasets = ["CIFAR10-S", "C-MNIST\n(BG)", "BFFHQ", "UTKFace"]
    van  = [73.22, 100.0, 60.27, 53.50]
    fair = [25.40, 8.45, 24.13, 38.83]
    cob  = [16.70, 7.46, 15.13, 35.00]

    x = np.arange(len(datasets)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    b1 = ax.bar(x - w, van,  w, label="Vanilla DD", color=C_VAN)
    b2 = ax.bar(x,     fair, w, label="FairDD", color=C_FAIR)
    b3 = ax.bar(x + w, cob,  w, label="COBRA", color=C_COB)
    for bars in (b1, b2, b3):
        _bar_labels(ax, bars, fs=12)
    ax.set_ylabel("EOD  $\\downarrow$")
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylim(0, 112)
    ax.set_title("COBRA gives the lowest unfairness (DM, IPC=50)",
                 fontsize=17, fontweight="bold", pad=12)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=13)
    _save(fig, "fig2_methods")


# ====================== FIG 3 : interaction redesign (clean 2-panel) ========= #
def fig3_interaction():
    # CIFAR10-S, DM, IPC = 50 (Table 2). 3 bold series only.
    skew = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    s_van = [27.05, 33.13, 43.53, 54.59, 61.94, 78.54]
    s_cob = [8.53, 8.99, 10.83, 11.19, 12.59, 13.27]
    s_full = [14.13, 19.13, 26.27, 33.90, 42.73, 52.60]

    gap = [0, 1, 2, 3, 4]
    g_van = [0.0, 46.13, 62.23, 72.40, 74.90]
    g_cob = [0.0, 8.42, 9.10, 17.28, 19.52]
    g_full = [0.0, 27.30, 39.50, 54.85, 59.00]

    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.8), sharey=True)
    kw = dict(lw=3.2, ms=9)
    for ax, xs, v, c, f, xlabel in [
        (axes[0], skew, s_van, s_cob, s_full, "Group Imbalance"),
        (axes[1], gap,  g_van, g_cob, g_full, "Representation Difference Grade"),
    ]:
        ax.plot(xs, v, "-o", color=C_VAN, label="Synthetic data", **kw)
        ax.plot(xs, f, "-X", color=C_FULL, label="Original data", **kw)
        ax.plot(xs, c, "-D", color=C_COB, label="Ours", **kw)
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Bias (Equalized Odds)", fontweight="bold")
    axes[0].legend(loc="upper left", fontsize=13)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.18)
    _save(fig, "fig3_interaction")


# ====================== FIG 4 : dumbbell (Vanilla -> COBRA drop) ============= #
def fig4_dumbbell():
    # DM, IPC = 100, EOD. Sorted by COBRA value.
    rows = [
        ("C-MNIST (BG)", 100.0, 7.58),
        ("C-FMNIST (BG)", 100.0, 22.40),
        ("CIFAR10-S", 82.87, 9.37),
        ("C-FMNIST (FG)", 100.0, 24.17),
        ("BFFHQ", 63.47, 7.87),
        ("UTKFace", 48.83, 32.33),
    ]
    rows = sorted(rows, key=lambda r: r[1])           # order by vanilla
    labels = [r[0] for r in rows]
    van = [r[1] for r in rows]; cob = [r[2] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for yi, v, c in zip(y, van, cob):
        ax.plot([c, v], [yi, yi], color="0.7", lw=3, zorder=1)
        ax.annotate("", xy=(c, yi), xytext=(v, yi),
                    arrowprops=dict(arrowstyle="-|>", color="0.55", lw=0))
    ax.scatter(van, y, s=170, color=C_VAN, zorder=3, label="Vanilla DD")
    ax.scatter(cob, y, s=170, color=C_COB, zorder=3, label="COBRA")
    for yi, v, c in zip(y, van, cob):
        ax.text(v + 2, yi, f"{v:.0f}", va="center", fontsize=12, color=C_VAN, fontweight="bold")
        ax.text(c - 2, yi, f"{c:.0f}", va="center", ha="right", fontsize=12, color=C_COB, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("EOD  $\\downarrow$", fontweight="bold")
    ax.set_xlim(-4, 112)
    ax.set_title("COBRA collapses the EOD gap (DM, IPC=100)",
                 fontsize=17, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=13)
    _save(fig, "fig4_dumbbell")


# ====================== FIG 5 : EOD vs IPC (compression makes it worse) ====== #
def fig5_vs_ipc():
    # CIFAR10-S, DM. EOD vs IPC.
    ipc = [10, 50, 100]
    van  = [56.25, 73.22, 82.87]
    fair = [25.58, 25.40, 25.17]
    cob  = [20.18, 16.70, 9.37]
    full = 48.96

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.axhline(full, ls=(0, (4, 3)), color=C_FULL, lw=2.2, label="Full data")
    ax.plot(ipc, van,  "-o", color=C_VAN,  lw=3.2, ms=10, label="Vanilla DD")
    ax.plot(ipc, fair, "-s", color=C_FAIR, lw=3.2, ms=10, label="FairDD")
    ax.plot(ipc, cob,  "-D", color=C_COB,  lw=3.2, ms=10, label="COBRA")
    ax.set_xticks(ipc)
    ax.set_xlabel("Images per class (IPC)", fontweight="bold")
    ax.set_ylabel("EOD  $\\downarrow$", fontweight="bold")
    ax.set_ylim(0, 92)
    ax.set_title("More compression $\\to$ vanilla worsens, COBRA improves\n(CIFAR10-S, DM)",
                 fontsize=16, fontweight="bold", pad=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=13, loc="center right")
    _save(fig, "fig5_vs_ipc")


if __name__ == "__main__":
    # fig1_amplification()
    # fig2_methods()
    fig3_interaction()
    # fig4_dumbbell()
    # fig5_vs_ipc()