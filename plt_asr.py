import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------

BUDGETS = np.array([5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1])
BUDGET_LABELS = [
    r"$5{\times}10^{-4}$",
    r"$10^{-3}$",
    r"$2{\times}10^{-3}$",
    r"$5{\times}10^{-3}$",
    r"$10^{-2}$",
    r"$2{\times}10^{-2}$",
    r"$5{\times}10^{-2}$",
    r"$10^{-1}$",
]

NA = np.nan

rows = []

def add(metric, model, pair, attack, variant, values):
    rows.append({
        "metric": metric,
        "model": model,
        "pair": pair,
        "attack": attack,
        "variant": variant,
        "values": values,
    })


# -----------------------------
# Table 2: CTA
# -----------------------------

add("CTA", "ConvNetBN", "dog-bird", "fc", "smart",
    [80.26, 80.33, 80.14, 80.27, 80.32, 80.28, 79.91, 75.33])
add("CTA", "ConvNetBN", "dog-bird", "fc", "random",
    [80.30, 80.20, 80.25, 80.33, 80.34, 80.29, 79.83, 75.51])
add("CTA", "ConvNetBN", "dog-bird", "gradmatch", "smart",
    [79.93, 80.22, 80.22, 80.18, 80.11, 80.08, 79.82, 76.65])
add("CTA", "ConvNetBN", "dog-bird", "gradmatch", "random",
    [80.13, 79.94, 80.31, 79.99, 80.22, 80.11, 79.94, 76.81])

add("CTA", "ConvNetBN", "frog-airplane", "fc", "smart",
    [NA, NA, NA, NA, 80.16, 79.89, 79.67, 74.44])
add("CTA", "ConvNetBN", "frog-airplane", "fc", "random",
    [NA, NA, NA, NA, 80.16, 80.24, 79.84, 74.25])
add("CTA", "ConvNetBN", "frog-airplane", "gradmatch", "smart",
    [NA, NA, 80.28, 80.26, 80.35, 80.31, 79.56, 76.77])
add("CTA", "ConvNetBN", "frog-airplane", "gradmatch", "random",
    [NA, NA, 80.43, 80.13, 80.17, 80.07, 80.19, 76.63])

add("CTA", "ResNet20", "dog-bird", "fc", "smart",
    [83.67, 83.69, 83.63, 83.70, 83.64, 83.37, 82.64, 77.47])
add("CTA", "ResNet20", "dog-bird", "fc", "random",
    [83.69, 83.68, 83.66, 83.55, 83.63, 83.47, 83.13, 77.57])
add("CTA", "ResNet20", "dog-bird", "gradmatch", "smart",
    [83.83, 83.80, 83.83, 83.80, 83.75, 83.67, 83.16, 77.95])
add("CTA", "ResNet20", "dog-bird", "gradmatch", "random",
    [83.76, 83.79, 83.90, 83.77, 83.83, 83.65, 83.38, 77.94])

add("CTA", "VGG13", "dog-bird", "fc", "smart",
    [85.51, 84.92, 84.94, 85.31, 85.15, 85.03, 83.86, 78.59])
add("CTA", "VGG13", "dog-bird", "fc", "random",
    [85.24, 85.28, 85.15, 85.05, 85.14, 85.09, 84.64, 78.65])
add("CTA", "VGG13", "dog-bird", "gradmatch", "smart",
    [85.26, 85.33, 85.50, 85.41, 85.21, 85.40, 84.25, 79.71])
add("CTA", "VGG13", "dog-bird", "gradmatch", "random",
    [85.54, 85.37, 85.41, 85.35, 85.24, 85.12, 84.88, 79.28])


# -----------------------------
# Table 3: ASR
# -----------------------------

add("ASR", "ConvNetBN", "dog-bird", "fc", "smart",
    [2.0, 4.0, 16.0, 22.0, 36.0, 48.0, 56.0, 62.0])
add("ASR", "ConvNetBN", "dog-bird", "fc", "random",
    [0.0, 4.0, 4.0, 8.0, 6.0, 16.0, 42.0, 76.0])
add("ASR", "ConvNetBN", "dog-bird", "gradmatch", "smart",
    [4.0, 8.0, 16.0, 32.0, 52.0, 88.0, 88.0, 98.0])
add("ASR", "ConvNetBN", "dog-bird", "gradmatch", "random",
    [0.0, 4.0, 6.0, 14.0, 12.0, 22.0, 72.0, 94.0])

add("ASR", "ConvNetBN", "frog-airplane", "fc", "smart",
    [NA, NA, NA, NA, 6.0, 8.0, 22.0, 26.0])
add("ASR", "ConvNetBN", "frog-airplane", "fc", "random",
    [NA, NA, NA, NA, 0.0, 0.0, 10.0, 26.0])
add("ASR", "ConvNetBN", "frog-airplane", "gradmatch", "smart",
    [NA, NA, 0.0, 0.0, 2.0, 6.0, 14.0, 20.0])
add("ASR", "ConvNetBN", "frog-airplane", "gradmatch", "random",
    [NA, NA, 0.0, 0.0, 0.0, 0.0, 8.0, 18.0])

add("ASR", "ResNet20", "dog-bird", "fc", "smart",
    [2.0, 2.0, 2.0, 2.0, 2.0, 14.0, 6.0, 8.0])
add("ASR", "ResNet20", "dog-bird", "fc", "random",
    [0.0, 2.0, 4.0, 2.0, 0.0, 2.0, 8.0, 6.0])
add("ASR", "ResNet20", "dog-bird", "gradmatch", "smart",
    [0.0, 0.0, 8.0, 16.0, 36.0, 32.0, 34.0, 14.0])
add("ASR", "ResNet20", "dog-bird", "gradmatch", "random",
    [2.0, 0.0, 2.0, 4.0, 8.0, 28.0, 46.0, 6.0])

add("ASR", "VGG13", "dog-bird", "fc", "smart",
    [2.0, 10.0, 20.0, 34.0, 34.0, 32.0, 10.0, 14.0])
add("ASR", "VGG13", "dog-bird", "fc", "random",
    [2.0, 6.0, 2.0, 10.0, 12.0, 16.0, 12.0, 12.0])
add("ASR", "VGG13", "dog-bird", "gradmatch", "smart",
    [48.0, 62.0, 82.0, 92.0, 94.0, 92.0, 96.0, 100.0])
add("ASR", "VGG13", "dog-bird", "gradmatch", "random",
    [4.0, 12.0, 22.0, 58.0, 80.0, 88.0, 98.0, 98.0])


# -----------------------------
# Convert to long dataframe
# -----------------------------

long_rows = []

for r in rows:
    for budget, score in zip(BUDGETS, r["values"]):
        long_rows.append({
            "metric": r["metric"],
            "model": r["model"],
            "pair": r["pair"],
            "attack": r["attack"],
            "variant": r["variant"],
            "budget": budget,
            "score": score,
        })

df = pd.DataFrame(long_rows)


# -----------------------------
# Plotting functions
# -----------------------------

def plot_grid(metric, pair, output_name):
    models = ["ConvNetBN", "ResNet20", "VGG13"]
    attacks = ["fc", "gradmatch"]

    fig, axes = plt.subplots(
        nrows=len(models),
        ncols=len(attacks),
        figsize=(10, 8),
        sharex=True,
        sharey=True,
    )

    for i, model in enumerate(models):
        for j, attack in enumerate(attacks):
            ax = axes[i, j]

            sub = df[
                (df["metric"] == metric)
                & (df["pair"] == pair)
                & (df["model"] == model)
                & (df["attack"] == attack)
            ]

            for variant, marker, linestyle in [
                ("smart", "o", "-"),
                ("random", "s", "--"),
            ]:
                s = sub[sub["variant"] == variant].sort_values("budget")

                if len(s) == 0 or s["score"].isna().all():
                    continue

                ax.plot(
                    s["budget"],
                    s["score"],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2,
                    markersize=5,
                    label=variant,
                )

            ax.set_xscale("log")
            ax.set_xticks(BUDGETS)
            ax.set_xticklabels(BUDGET_LABELS, rotation=35, ha="right")
            ax.grid(True, linewidth=0.5, alpha=0.4)

            if i == 0:
                ax.set_title(attack)
            if j == 0:
                ax.set_ylabel(f"{model}\n{metric} (%)")
            if i == len(models) - 1:
                ax.set_xlabel(r"Perturbation budget $\epsilon$")

            if sub.empty or sub["score"].isna().all():
                ax.text(
                    0.5,
                    0.5,
                    "not run",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                )

    if metric == "ASR":
        axes[0, 0].set_ylim(-2, 105)

    fig.suptitle(f"{metric} on {pair}", fontsize=16)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.96))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{output_name}.pdf", bbox_inches="tight")
    plt.savefig(f"{output_name}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_asr_cta_overlay(model, pair, attack, output_name):
    fig, ax1 = plt.subplots(figsize=(7, 4))

    for variant, marker, linestyle in [
        ("smart", "o", "-"),
        ("random", "s", "--"),
    ]:
        sub_asr = df[
            (df["metric"] == "ASR")
            & (df["model"] == model)
            & (df["pair"] == pair)
            & (df["attack"] == attack)
            & (df["variant"] == variant)
        ].sort_values("budget")

        ax1.plot(
            sub_asr["budget"],
            sub_asr["score"],
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            label=f"ASR {variant}",
        )

    ax1.set_xscale("log")
    ax1.set_xticks(BUDGETS)
    ax1.set_xticklabels(BUDGET_LABELS, rotation=35, ha="right")
    ax1.set_xlabel(r"Perturbation budget $\epsilon$")
    ax1.set_ylabel("ASR (%)")
    ax1.set_ylim(-2, 105)
    ax1.grid(True, linewidth=0.5, alpha=0.4)

    ax2 = ax1.twinx()

    for variant, marker, linestyle in [
        ("smart", "^", ":"),
        ("random", "v", "-."),
    ]:
        sub_cta = df[
            (df["metric"] == "CTA")
            & (df["model"] == model)
            & (df["pair"] == pair)
            & (df["attack"] == attack)
            & (df["variant"] == variant)
        ].sort_values("budget")

        ax2.plot(
            sub_cta["budget"],
            sub_cta["score"],
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            label=f"CTA {variant}",
        )

    ax2.set_ylabel("CTA (%)")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.title(f"{model}, {pair}, {attack}")
    plt.tight_layout()
    plt.savefig(f"{output_name}.pdf", bbox_inches="tight")
    plt.savefig(f"{output_name}.png", dpi=300, bbox_inches="tight")
    plt.show()


# -----------------------------
# Make plots
# -----------------------------

plot_grid("ASR", "dog-bird", "asr_dog_bird")
plot_grid("CTA", "dog-bird", "cta_dog_bird")

plot_grid("ASR", "frog-airplane", "asr_frog_airplane")
plot_grid("CTA", "frog-airplane", "cta_frog_airplane")

# Best single-result overlays
plot_asr_cta_overlay(
    model="VGG13",
    pair="dog-bird",
    attack="gradmatch",
    output_name="vgg13_dog_bird_gradmatch_overlay",
)

plot_asr_cta_overlay(
    model="ConvNetBN",
    pair="dog-bird",
    attack="gradmatch",
    output_name="convnetbn_dog_bird_gradmatch_overlay",
)