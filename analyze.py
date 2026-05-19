"""
实验完成后运行：.venv/bin/python analyze.py
生成所有图表，保存到 results/figures/
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

METRICS_DIR = "results/metrics"
FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

AUGMENT_NAMES  = ["none", "flip_crop", "cutout", "mixup", "color_jitter"]
AUGMENT_LABELS = ["None", "Flip+Crop", "Cutout", "MixUp", "Color Jitter"]
AUGMENT_COLORS = ["#2c3e50", "#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
AUGMENT_STYLES = ["-", "--", "-.", ":", (0,(3,1,1,1))]


def load_results(exp_filter=None, model_filter=None, dataset_filter=None):
    results = []
    for fname in os.listdir(METRICS_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(METRICS_DIR, fname)) as f:
            d = json.load(f)
        if d.get("status") != "completed":
            continue
        if exp_filter and d.get("exp") != exp_filter:
            continue
        if model_filter and d.get("model") != model_filter:
            continue
        if dataset_filter and d.get("dataset") != dataset_filter:
            continue
        results.append(d)
    return results


def aggregate(results, group_keys):
    """按 group_keys 分组，取 best_test_error 的均值和标准差"""
    groups = defaultdict(list)
    for r in results:
        key = tuple(r[k] for k in group_keys)
        groups[key].append(r["best_test_error"])
    agg = {}
    for key, vals in groups.items():
        agg[key] = (np.mean(vals), np.std(vals))
    return agg


# ── Fig 1/2：Double Descent 曲线 ──────────────────────────────────────

def plot_double_descent(model_name: str, fig_path: str):
    results = load_results(exp_filter="exp1", model_filter=model_name, dataset_filter="cifar10")
    if not results:
        print(f"No results for {model_name}, skipping.")
        return

    agg = aggregate(results, ["augment", "width_idx"])
    width_idxs = sorted(set(r["width_idx"] for r in results))
    n_params    = {}
    for r in results:
        n_params[r["width_idx"]] = r["n_params"]
    x_params = [n_params[i] for i in width_idxs]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for aug, label, color, style in zip(AUGMENT_NAMES, AUGMENT_LABELS, AUGMENT_COLORS, AUGMENT_STYLES):
        means, stds = [], []
        for wi in width_idxs:
            key = (aug, wi)
            if key in agg:
                m, s = agg[key]
            else:
                m, s = np.nan, 0
            means.append(m)
            stds.append(s)
        means, stds = np.array(means), np.array(stds)
        ax.plot(range(len(width_idxs)), means, label=label, color=color,
                linestyle=style, linewidth=2, marker="o", markersize=4)
        ax.fill_between(range(len(width_idxs)),
                        means - stds, means + stds,
                        alpha=0.12, color=color)

    ax.set_xticks(range(len(width_idxs)))
    ax.set_xticklabels([f"{n_params.get(i,0)/1e3:.0f}K" for i in width_idxs],
                       rotation=30, fontsize=8)
    ax.set_xlabel("Model Size (parameters)", fontsize=11)
    ax.set_ylabel("Test Error", fontsize=11)
    title = "MLP" if model_name == "mlp" else "ResNet"
    ax.set_title(f"Double Descent under Different Augmentations ({title}, CIFAR-10)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # 标记插值阈值区域（无增强的峰值位置）
    none_means = [agg.get(("none", wi), (np.nan,))[0] for wi in width_idxs]
    if not all(np.isnan(none_means)):
        peak_idx = int(np.nanargmax(none_means))
        ax.axvline(peak_idx, color="gray", linestyle=":", alpha=0.5, linewidth=1)
        ax.text(peak_idx + 0.1, ax.get_ylim()[1] * 0.95,
                "interpolation\nthreshold\n(no aug)", fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


# ── Fig 3：阈值位置对比柱状图 ─────────────────────────────────────────

def plot_threshold_comparison(fig_path: str):
    thresholds = {}
    for model_name in ["mlp", "resnet"]:
        results = load_results(exp_filter="exp1", model_filter=model_name, dataset_filter="cifar10")
        if not results:
            continue
        agg = aggregate(results, ["augment", "width_idx"])
        width_idxs = sorted(set(r["width_idx"] for r in results))
        for aug in AUGMENT_NAMES:
            means = [agg.get((aug, wi), (np.nan,))[0] for wi in width_idxs]
            if not all(np.isnan(means)):
                peak_idx = int(np.nanargmax(means))
                thresholds[(model_name, aug)] = peak_idx

    if not thresholds:
        print("No threshold data, skipping Fig 3.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, model_name in zip(axes, ["mlp", "resnet"]):
        vals = [thresholds.get((model_name, aug), 0) for aug in AUGMENT_NAMES]
        bars = ax.bar(AUGMENT_LABELS, vals, color=AUGMENT_COLORS, alpha=0.85, edgecolor="white")
        ax.set_ylabel("Threshold Width Index (higher = more params)", fontsize=9)
        ax.set_title(f"Interpolation Threshold — {'MLP' if model_name=='mlp' else 'ResNet'}",
                     fontsize=10)
        ax.set_ylim(0, len(set(r["width_idx"] for r in load_results("exp1", model_name))))
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    str(val), ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Effect of Augmentation on Double Descent Threshold Position", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


# ── Fig 4：MixUp 强度 ablation ────────────────────────────────────────

def plot_mixup_ablation(fig_path: str):
    results = load_results(exp_filter="exp3")
    if not results:
        print("No exp3 results, skipping Fig 4.")
        return

    alphas    = sorted(set(r.get("mixup_alpha", 0.4) for r in results))
    width_idxs = sorted(set(r["width_idx"] for r in results))

    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(alphas)-1, 1)) for i in range(len(alphas))]

    for alpha, color in zip(alphas, colors):
        r_alpha = [r for r in results if abs(r.get("mixup_alpha", 0.4) - alpha) < 0.01]
        agg = aggregate(r_alpha, ["width_idx"])
        means = [agg.get((wi,), (np.nan,))[0] for wi in width_idxs]
        ax.plot(range(len(width_idxs)), means, label=f"α={alpha}",
                color=color, linewidth=2, marker="o", markersize=4)

    ax.set_xticks(range(len(width_idxs)))
    ax.set_xticklabels([f"w{i}" for i in width_idxs], fontsize=8)
    ax.set_xlabel("Model Width Index", fontsize=11)
    ax.set_ylabel("Test Error", fontsize=11)
    ax.set_title("MixUp Intensity (α) vs Double Descent (MLP, CIFAR-10)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


# ── Table 1：CIFAR-100 泛化结果 ───────────────────────────────────────

def generate_table(table_path: str):
    results = load_results(exp_filter="exp4", model_filter="resnet", dataset_filter="cifar100")
    if not results:
        print("No exp4 results, skipping Table 1.")
        return

    agg = aggregate(results, ["augment", "width_idx"])
    width_idxs = sorted(set(r["width_idx"] for r in results))

    lines = ["% Table 1: CIFAR-100 Generalization (ResNet)",
             "\\begin{table}[h]",
             "\\centering",
             "\\caption{Test error on CIFAR-100 across augmentation strategies (ResNet).}",
             "\\begin{tabular}{l" + "c" * len(width_idxs) + "}",
             "\\hline",
             "Augmentation & " + " & ".join([f"w{i}" for i in width_idxs]) + " \\\\",
             "\\hline"]

    for aug, label in zip(["none", "mixup", "cutout"], ["None", "MixUp", "Cutout"]):
        row = [label]
        for wi in width_idxs:
            val = agg.get((aug, wi), (np.nan,))[0]
            row.append(f"{val:.3f}" if not np.isnan(val) else "--")
        lines.append(" & ".join(row) + " \\\\")

    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    with open(table_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {table_path}")


# ── 主入口 ────────────────────────────────────────────────────────────

def main():
    print("生成所有图表...")
    plot_double_descent("mlp",    f"{FIGURES_DIR}/fig1_dd_mlp.pdf")
    plot_double_descent("resnet", f"{FIGURES_DIR}/fig2_dd_resnet.pdf")
    plot_threshold_comparison(    f"{FIGURES_DIR}/fig3_threshold.pdf")
    plot_mixup_ablation(          f"{FIGURES_DIR}/fig4_mixup_ablation.pdf")
    generate_table(               f"{FIGURES_DIR}/table1_cifar100.tex")
    print("\n所有图表生成完成！保存在 results/figures/")


if __name__ == "__main__":
    main()
