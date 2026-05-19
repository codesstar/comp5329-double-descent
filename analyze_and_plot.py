"""
Complete analysis and figure generation for double descent + data augmentation study.
"""

import json
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────────
METRICS_DIR = "/Users/callum/Desktop/homework/results/metrics"
FIGURES_DIR = "/Users/callum/Desktop/homework/results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
AUG_ORDER  = ["none", "flip_crop", "cutout", "mixup", "color_jitter"]
AUG_LABELS = {
    "none":         "None",
    "flip_crop":    "Flip+Crop",
    "cutout":       "Cutout",
    "mixup":        "MixUp",
    "color_jitter": "Color Jitter",
}
AUG_COLORS = {
    "none":         "#e41a1c",
    "flip_crop":    "#377eb8",
    "cutout":       "#4daf4a",
    "mixup":        "#984ea3",
    "color_jitter": "#ff7f00",
}
AUG_MARKERS = {
    "none":         "o",
    "flip_crop":    "s",
    "cutout":       "^",
    "mixup":        "D",
    "color_jitter": "v",
}

# ResNet width_idx → n_params mapping (from actual data)
RESNET_PARAMS = {0: 11374, 1: 44370, 2: 175258, 3: 392674,
                 4: 696618, 5: 1564090, 6: 2777674, 7: 6243178}

# MLP width_idx → n_params mapping
MLP_PARAMS = {
    -3: 6e3, -2: 12e3, -1: 25e3,
     0: 50e3,  1: 100e3, 2: 201e3, 3: 411e3,
     4: 855e3, 5: 1.84e6, 6: 4.21e6, 7: 10.5e6,
}

# ── load all JSON files ────────────────────────────────────────────────────────
def load_all():
    records = []
    for fname in os.listdir(METRICS_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(METRICS_DIR, fname)
        with open(path) as f:
            d = json.load(f)
        records.append(d)
    return records

records = load_all()
print(f"Loaded {len(records)} records total")

# ── helper: group by key ───────────────────────────────────────────────────────
def group_records(records, exp=None, model=None):
    out = []
    for r in records:
        if exp   and r.get("exp")   != exp:   continue
        if model and r.get("model") != model: continue
        out.append(r)
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 – Data Analysis
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("SECTION 1: MLP — test_error by augmentation × width_idx")
print("="*70)

exp1_mlp = group_records(records, exp="exp1", model="mlp")

# Build dict: aug -> width_idx -> list[test_error]
mlp_data = defaultdict(lambda: defaultdict(list))
for r in exp1_mlp:
    aug = r["augment"]
    # width_idx may be encoded as number (some runs use 'wx2' etc in filename but
    # the json should have numeric width_idx)
    wi = r.get("width_idx")
    if wi is None:
        continue
    mlp_data[aug][wi].append(r["test_error"])

# Collect all width indices
all_mlp_wi = sorted(set(wi for aug in mlp_data for wi in mlp_data[aug]))
print(f"MLP width indices present: {all_mlp_wi}")

mlp_summary = {}   # aug -> {wi: (mean, std, n)}
for aug in AUG_ORDER:
    if aug not in mlp_data:
        print(f"  WARNING: aug '{aug}' missing from MLP data!")
        continue
    mlp_summary[aug] = {}
    for wi in all_mlp_wi:
        vals = mlp_data[aug][wi]
        if not vals:
            continue
        mlp_summary[aug][wi] = (np.mean(vals), np.std(vals), len(vals))

print("\nMLP mean test_error table (aug × width_idx):")
header = f"{'aug':15s}" + "".join(f"  wi={wi:3d}" for wi in all_mlp_wi)
print(header)
for aug in AUG_ORDER:
    if aug not in mlp_summary:
        continue
    row = f"{AUG_LABELS[aug]:15s}"
    for wi in all_mlp_wi:
        if wi in mlp_summary[aug]:
            m, s, n = mlp_summary[aug][wi]
            row += f"  {m:.4f}"
        else:
            row += "       -"
    print(row)

# ── ResNet ─────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 2: ResNet — test_error by augmentation × width_idx")
print("="*70)

exp1_resnet = group_records(records, exp="exp1", model="resnet")

resnet_data = defaultdict(lambda: defaultdict(list))
for r in exp1_resnet:
    aug = r["augment"]
    wi  = r.get("width_idx")
    if wi is None:
        continue
    resnet_data[aug][wi].append(r["test_error"])

all_rn_wi = sorted(set(wi for aug in resnet_data for wi in resnet_data[aug]))
print(f"ResNet width indices present: {all_rn_wi}")

resnet_summary = {}
for aug in AUG_ORDER:
    if aug not in resnet_data:
        print(f"  WARNING: aug '{aug}' missing from ResNet data!")
        continue
    resnet_summary[aug] = {}
    for wi in all_rn_wi:
        vals = resnet_data[aug][wi]
        if not vals:
            continue
        resnet_summary[aug][wi] = (np.mean(vals), np.std(vals), len(vals))

print("\nResNet mean test_error table (aug × width_idx → params):")
param_labels = [f"w{wi}\n{RESNET_PARAMS[wi]/1e3:.0f}K" for wi in all_rn_wi]
header = f"{'aug':15s}" + "".join(f"  w{wi}({RESNET_PARAMS[wi]/1e3:.0f}K)" for wi in all_rn_wi)
print(header)
for aug in AUG_ORDER:
    if aug not in resnet_summary:
        continue
    row = f"{AUG_LABELS[aug]:15s}"
    for wi in all_rn_wi:
        if wi in resnet_summary[aug]:
            m, s, n = resnet_summary[aug][wi]
            row += f"  {m:.4f}"
        else:
            row += "         -"
    print(row)

# ── Find double-descent peaks for ResNet ──────────────────────────────────────
print("\n" + "="*70)
print("SECTION 2b: ResNet double descent peak analysis")
print("="*70)

peak_info = {}   # aug -> (peak_wi, peak_params, peak_error, valley_wi)

for aug in AUG_ORDER:
    if aug not in resnet_summary:
        continue
    wis    = sorted(resnet_summary[aug].keys())
    errors = [resnet_summary[aug][wi][0] for wi in wis]

    # Find the double-descent "interpolation threshold peak":
    # Classic shape: error decreases monotonically until near-interpolation,
    # then peaks, then decreases again (or stays high).
    # Strategy:
    #   1. Find global minimum (the "well-specified" region trough)
    #   2. Look for local maximum AFTER the minimum (this is the DD peak)
    #   3. If no local max after min, use max in the range [min_idx+1, end]
    #   Fallback: if monotone decreasing everywhere, report argmax overall.

    min_idx   = int(np.argmin(errors))
    valley_wi = wis[min_idx]

    # Look for a local max strictly after the minimum
    post_errors = errors[min_idx:]
    post_wis    = wis[min_idx:]
    dd_peak_idx_local = int(np.argmax(post_errors))
    dd_peak_wi    = post_wis[dd_peak_idx_local]
    dd_peak_error = post_errors[dd_peak_idx_local]
    dd_peak_params = RESNET_PARAMS.get(dd_peak_wi, float('nan'))

    # Only call it a genuine DD peak if the peak is meaningfully higher than min
    # AND is not the very first point (i.e., curve actually goes up after the min)
    dd_magnitude = dd_peak_error - errors[min_idx]

    if dd_peak_wi == valley_wi or dd_magnitude < 0.002:
        # No clear post-min rise → curve is monotone; fall back to global argmax
        peak_idx    = int(np.argmax(errors))
        peak_wi     = wis[peak_idx]
        peak_error  = errors[peak_idx]
        peak_params = RESNET_PARAMS.get(peak_wi, float('nan'))
    else:
        peak_wi     = dd_peak_wi
        peak_error  = dd_peak_error
        peak_params = dd_peak_params

    peak_info[aug] = {
        "peak_wi": peak_wi,
        "peak_params": peak_params,
        "peak_error": peak_error,
        "valley_wi": valley_wi,
        "wis": wis,
        "errors": errors,
    }

    print(f"\n  [{AUG_LABELS[aug]}]")
    for wi, err in zip(wis, errors):
        std = resnet_summary[aug][wi][1]
        n   = resnet_summary[aug][wi][2]
        marker = " ← PEAK" if wi == peak_wi else ""
        print(f"    w{wi} ({RESNET_PARAMS[wi]/1e3:.0f}K params): err={err:.4f} ± {std:.4f} (n={n}){marker}")

# ── exp3: MixUp alpha ablation ─────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 3: exp3 — MixUp alpha ablation")
print("="*70)

exp3 = [r for r in records if r.get("exp") == "exp3"]
print(f"exp3 records: {len(exp3)}")

# Parse alpha from run_id: exp3_mixup_a0p1_w0_s0
alpha_map = {}
for r in exp3:
    run_id = r["run_id"]
    m = re.search(r"a(\d+p\d+)", run_id)
    if not m:
        m = re.search(r"a(\d+)", run_id)
    if not m:
        print(f"  Can't parse alpha from: {run_id}")
        continue
    alpha_str = m.group(1).replace("p", ".")
    alpha = float(alpha_str)
    wi    = r.get("width_idx")
    if wi is None:
        continue
    if alpha not in alpha_map:
        alpha_map[alpha] = defaultdict(list)
    alpha_map[alpha][wi].append(r["test_error"])

alphas = sorted(alpha_map.keys())
print(f"Alphas found: {alphas}")
all_exp3_wi = sorted(set(wi for a in alpha_map for wi in alpha_map[a]))
print(f"Width indices: {all_exp3_wi}")

exp3_summary = {}
for alpha in alphas:
    exp3_summary[alpha] = {}
    for wi in all_exp3_wi:
        vals = alpha_map[alpha][wi]
        if vals:
            exp3_summary[alpha][wi] = (np.mean(vals), np.std(vals), len(vals))

print("\nexp3 mean test_error (alpha × width_idx):")
header = f"{'alpha':8s}" + "".join(f"  wi={wi:3d}" for wi in all_exp3_wi)
print(header)
for alpha in alphas:
    row = f"{alpha:<8.1f}"
    for wi in all_exp3_wi:
        if wi in exp3_summary[alpha]:
            m, s, n = exp3_summary[alpha][wi]
            row += f"  {m:.4f}"
        else:
            row += "       -"
    print(row)

# Best alpha per width
print("\nBest alpha per width_idx:")
for wi in all_exp3_wi:
    best_alpha = None
    best_err   = float('inf')
    for alpha in alphas:
        if wi in exp3_summary[alpha]:
            err = exp3_summary[alpha][wi][0]
            if err < best_err:
                best_err   = err
                best_alpha = alpha
    params_label = f"{MLP_PARAMS.get(wi, RESNET_PARAMS.get(wi, 0))/1e3:.0f}K"
    print(f"  wi={wi} ({params_label} params): best alpha={best_alpha}, err={best_err:.4f}")

# Overall best alpha (averaged over all widths)
print("\nOverall mean test_error by alpha (averaged over all widths):")
for alpha in alphas:
    errs = [exp3_summary[alpha][wi][0] for wi in all_exp3_wi if wi in exp3_summary[alpha]]
    print(f"  alpha={alpha:.1f}: mean_err={np.mean(errs):.4f}")

# ── exp4: CIFAR-100 ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SECTION 4: exp4 — CIFAR-100 validation")
print("="*70)

exp4 = [r for r in records if r.get("exp") == "exp4"]
print(f"exp4 records: {len(exp4)}")
exp4_augs = sorted(set(r["augment"] for r in exp4))
print(f"Augmentations in exp4: {exp4_augs}")

exp4_data = defaultdict(lambda: defaultdict(list))
for r in exp4:
    exp4_data[r["augment"]][r["width_idx"]].append(r["test_error"])

all_exp4_wi = sorted(set(wi for aug in exp4_data for wi in exp4_data[aug]))
exp4_summary = {}
for aug in exp4_augs:
    exp4_summary[aug] = {}
    for wi in all_exp4_wi:
        vals = exp4_data[aug][wi]
        if vals:
            exp4_summary[aug][wi] = (np.mean(vals), np.std(vals), len(vals))

print("\nexp4 (CIFAR-100) mean test_error:")
header = f"{'aug':15s}" + "".join(f"  w{wi}({RESNET_PARAMS.get(wi,0)/1e3:.0f}K)" for wi in all_exp4_wi)
print(header)
for aug in exp4_augs:
    row = f"{aug:15s}"
    for wi in all_exp4_wi:
        if wi in exp4_summary.get(aug, {}):
            m, s, n = exp4_summary[aug][wi]
            row += f"  {m:.4f}"
        else:
            row += "          -"
    print(row)

print("\nBest (lowest) test_error per augmentation on CIFAR-100:")
for aug in exp4_augs:
    if aug not in exp4_summary:
        continue
    best_wi  = min(exp4_summary[aug], key=lambda wi: exp4_summary[aug][wi][0])
    best_err = exp4_summary[aug][best_wi][0]
    final_wi = max(exp4_summary[aug].keys())
    final_err = exp4_summary[aug][final_wi][0]
    print(f"  {aug:15s}: best={best_err:.4f} (w{best_wi}), largest_model={final_err:.4f} (w{final_wi})")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 – Figure generation
# ═══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":      150,
    "savefig.dpi":     150,
    "savefig.bbox":    "tight",
})

def params_to_M(p):
    if p >= 1e6:
        return f"{p/1e6:.2f}M"
    else:
        return f"{p/1e3:.0f}K"

# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: MLP double descent
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for aug in AUG_ORDER:
    if aug not in mlp_summary:
        continue
    wis    = sorted(mlp_summary[aug].keys())
    params = [MLP_PARAMS[wi] for wi in wis]
    means  = [mlp_summary[aug][wi][0] for wi in wis]
    stds   = [mlp_summary[aug][wi][1] for wi in wis]

    ax.plot(params, means,
            color=AUG_COLORS[aug], marker=AUG_MARKERS[aug],
            label=AUG_LABELS[aug], linewidth=1.8, markersize=5, zorder=3)
    ax.fill_between(params,
                    [m-s for m, s in zip(means, stds)],
                    [m+s for m, s in zip(means, stds)],
                    color=AUG_COLORS[aug], alpha=0.12)

ax.set_xscale("log")
ax.set_xlabel("Number of Parameters")
ax.set_ylabel("Test Error")
ax.set_title("Fig. 1  MLP: Test Error vs. Model Size (CIFAR-10)")
ax.legend(loc="upper right", framealpha=0.9)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: params_to_M(x)))
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xlim(left=4e3)

fig.savefig(os.path.join(FIGURES_DIR, "fig1_mlp_double_descent.png"))
plt.close(fig)
print("\nSaved: fig1_mlp_double_descent.png")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: ResNet double descent with peak annotations
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for aug in AUG_ORDER:
    if aug not in resnet_summary:
        continue
    wis    = sorted(resnet_summary[aug].keys())
    params = [RESNET_PARAMS[wi] for wi in wis]
    means  = [resnet_summary[aug][wi][0] for wi in wis]
    stds   = [resnet_summary[aug][wi][1] for wi in wis]

    ax.plot(params, means,
            color=AUG_COLORS[aug], marker=AUG_MARKERS[aug],
            label=AUG_LABELS[aug], linewidth=1.8, markersize=5, zorder=3)
    ax.fill_between(params,
                    [m-s for m, s in zip(means, stds)],
                    [m+s for m, s in zip(means, stds)],
                    color=AUG_COLORS[aug], alpha=0.12)

# Annotate double-descent peaks (use 'none' as primary example)
for aug in ["none"]:
    if aug not in peak_info:
        continue
    pi = peak_info[aug]
    px = RESNET_PARAMS[pi["peak_wi"]]
    py = pi["peak_error"]
    ax.annotate(
        f'DD peak\n({params_to_M(px)})',
        xy=(px, py),
        xytext=(px * 1.5, py + 0.008),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
        fontsize=8.5, color="#333333",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.8),
    )

ax.set_xscale("log")
ax.set_xlabel("Number of Parameters")
ax.set_ylabel("Test Error")
ax.set_title("Fig. 2  ResNet: Test Error vs. Model Size (CIFAR-10)\nDouble Descent with Different Augmentation Strategies")
ax.legend(loc="upper left", framealpha=0.9)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: params_to_M(x)))
ax.grid(True, alpha=0.3, linestyle="--")

fig.savefig(os.path.join(FIGURES_DIR, "fig2_resnet_double_descent.png"))
plt.close(fig)
print("Saved: fig2_resnet_double_descent.png")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Peak position + peak height comparison bar chart
# ──────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

augs_with_peaks = [aug for aug in AUG_ORDER if aug in peak_info]
peak_heights = [peak_info[aug]["peak_error"] for aug in augs_with_peaks]
peak_params  = [peak_info[aug]["peak_params"] / 1e6 for aug in augs_with_peaks]  # in M
colors = [AUG_COLORS[aug] for aug in augs_with_peaks]
xlabels = [AUG_LABELS[aug] for aug in augs_with_peaks]

# Bar 1: peak error height
bars1 = ax1.bar(xlabels, peak_heights, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
ax1.set_ylabel("Peak Test Error")
ax1.set_title("(a) Double Descent Peak Height\n(ResNet, CIFAR-10)")
ax1.set_ylim(0, max(peak_heights) * 1.2)
ax1.tick_params(axis='x', rotation=20)
for bar, val in zip(bars1, peak_heights):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f"{val:.4f}", ha='center', va='bottom', fontsize=8)

# Bar 2: peak position in log-scale (params in M)
bars2 = ax2.bar(xlabels, peak_params, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
ax2.set_ylabel("Peak Position (M params)")
ax2.set_title("(b) Double Descent Peak Position\n(ResNet, CIFAR-10)")
ax2.set_yscale("log")
ax2.tick_params(axis='x', rotation=20)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}M" if x >= 1 else f"{x*1000:.0f}K"))
for bar, val, aug in zip(bars2, peak_params, augs_with_peaks):
    wi = peak_info[aug]["peak_wi"]
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
             f"w{wi}\n{params_to_M(peak_info[aug]['peak_params'])}",
             ha='center', va='bottom', fontsize=7.5)

fig.suptitle("Fig. 3  ResNet Double Descent Peak Analysis by Augmentation", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig3_peak_comparison.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: fig3_peak_comparison.png")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 4: MixUp alpha ablation (exp3)
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Detect model type for exp3
sample_exp3_rec = [r for r in exp3 if r.get("width_idx") == 0]
model_type_exp3 = sample_exp3_rec[0].get("model", "mlp") if sample_exp3_rec else "mlp"
param_map_exp3 = MLP_PARAMS if model_type_exp3 == "mlp" else RESNET_PARAMS

# Left: error curves per alpha across widths — all overlap, demonstrating insensitivity
alpha_colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(alphas)))
ax = axes[0]
for alpha, color in zip(alphas, alpha_colors):
    if alpha not in exp3_summary:
        continue
    wis    = sorted(exp3_summary[alpha].keys())
    params = [param_map_exp3.get(wi, wi) for wi in wis]
    means  = [exp3_summary[alpha][wi][0] for wi in wis]
    stds   = [exp3_summary[alpha][wi][1] for wi in wis]

    ax.plot(params, means, color=color, marker="o", label=f"α={alpha:.1f}",
            linewidth=2.0, markersize=5, alpha=0.85)

ax.set_xscale("log")
ax.set_xlabel("Number of Parameters")
ax.set_ylabel("Test Error")
ax.set_title(f"(a) Test Error vs. Model Size by MixUp α\n({model_type_exp3.upper()}, CIFAR-10)")
ax.legend(loc="upper right", framealpha=0.9)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: params_to_M(x)))
ax.grid(True, alpha=0.3, linestyle="--")
ax.text(0.05, 0.15, "All α values nearly identical\n→ MixUp α-insensitive on MLP",
        transform=ax.transAxes, fontsize=8.5, color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffffcc", ec="#cccc88", alpha=0.9))

# Right: mean error vs alpha — flat line confirms insensitivity
ax2 = axes[1]
mean_errors_by_alpha = []
for alpha in alphas:
    errs = [exp3_summary[alpha][wi][0] for wi in all_exp3_wi if wi in exp3_summary.get(alpha, {})]
    mean_errors_by_alpha.append(np.mean(errs) if errs else float('nan'))

err_range = max(mean_errors_by_alpha) - min(mean_errors_by_alpha)
ax2.plot(alphas, mean_errors_by_alpha, marker="D", color="#984ea3",
         linewidth=2, markersize=8, zorder=3)
for alpha, err in zip(alphas, mean_errors_by_alpha):
    offset = max(err_range * 0.1, 0.0005)
    ax2.text(alpha, err + offset, f"{err:.4f}", ha="center", va="bottom", fontsize=8)
ax2.set_xlabel("MixUp α")
ax2.set_ylabel("Mean Test Error (all widths)")
ax2.set_title("(b) Mean Test Error vs. MixUp α\n(flat line → robust to α choice)")
ax2.grid(True, alpha=0.3, linestyle="--")
ax2.set_xticks(alphas)
y_center = np.nanmean(mean_errors_by_alpha)
y_pad = max(err_range * 3, 0.005)
ax2.set_ylim(y_center - y_pad, y_center + y_pad)

fig.suptitle("Fig. 4  MixUp α Ablation Study (MLP, CIFAR-10)", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig4_mixup_ablation.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: fig4_mixup_ablation.png")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 5: Augmentation ranking heatmap (ResNet)
# ──────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Build error matrix: aug × width_idx
augs_in_resnet = [aug for aug in AUG_ORDER if aug in resnet_summary]
rn_wis = sorted(set(wi for aug in augs_in_resnet for wi in resnet_summary[aug]))
err_matrix = np.full((len(augs_in_resnet), len(rn_wis)), np.nan)
for i, aug in enumerate(augs_in_resnet):
    for j, wi in enumerate(rn_wis):
        if wi in resnet_summary[aug]:
            err_matrix[i, j] = resnet_summary[aug][wi][0]

# Rank matrix (1=best at each width)
rank_matrix = np.zeros_like(err_matrix)
for j in range(err_matrix.shape[1]):
    col = err_matrix[:, j]
    valid = ~np.isnan(col)
    ranks = np.empty_like(col)
    ranks[valid]  = np.argsort(np.argsort(col[valid])) + 1
    ranks[~valid] = np.nan
    rank_matrix[:, j] = ranks

im1 = ax1.imshow(rank_matrix, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=len(augs_in_resnet))
ax1.set_xticks(range(len(rn_wis)))
ax1.set_xticklabels([f"w{wi}\n{params_to_M(RESNET_PARAMS[wi])}" for wi in rn_wis], fontsize=8)
ax1.set_yticks(range(len(augs_in_resnet)))
ax1.set_yticklabels([AUG_LABELS[aug] for aug in augs_in_resnet])
ax1.set_title("(a) Rank (1=best) by Width — ResNet")
for i in range(len(augs_in_resnet)):
    for j in range(len(rn_wis)):
        val = rank_matrix[i, j]
        if not np.isnan(val):
            ax1.text(j, i, f"{int(val)}", ha="center", va="center",
                     fontsize=9, color="black", fontweight="bold")
        else:
            ax1.text(j, i, "—", ha="center", va="center",
                     fontsize=9, color="#aaaaaa")
plt.colorbar(im1, ax=ax1, label="Rank")

# Right: line plot of error vs params for each aug (ResNet)
for aug in augs_in_resnet:
    wis    = sorted(resnet_summary[aug].keys())
    params = [RESNET_PARAMS[wi] for wi in wis]
    means  = [resnet_summary[aug][wi][0] for wi in wis]
    ax2.plot(params, means, color=AUG_COLORS[aug], marker=AUG_MARKERS[aug],
             label=AUG_LABELS[aug], linewidth=1.8, markersize=5)

ax2.set_xscale("log")
ax2.set_xlabel("Number of Parameters")
ax2.set_ylabel("Test Error")
ax2.set_title("(b) Test Error Comparison — ResNet")
ax2.legend(loc="upper left", framealpha=0.9)
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: params_to_M(x)))
ax2.grid(True, alpha=0.3, linestyle="--")

fig.suptitle("Fig. 5  Augmentation Strategy Comparison (ResNet, CIFAR-10)", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "fig5_augmentation_ranking.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: fig5_augmentation_ranking.png")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 6: CIFAR-100 table figure
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.axis("off")

# Build table data
exp4_augs_sorted = sorted(exp4_augs, key=lambda a: AUG_ORDER.index(a) if a in AUG_ORDER else 99)
exp4_wis_sorted  = sorted(all_exp4_wi)

col_labels = ["Augmentation"] + [f"w{wi}\n({params_to_M(RESNET_PARAMS.get(wi,0))})" for wi in exp4_wis_sorted] + ["Best"]
table_data = []
row_colors = []

best_per_aug = {}
for aug in exp4_augs_sorted:
    row = [AUG_LABELS.get(aug, aug)]
    errs_this_aug = []
    for wi in exp4_wis_sorted:
        if wi in exp4_summary.get(aug, {}):
            m, s, n = exp4_summary[aug][wi]
            row.append(f"{m:.4f}")
            errs_this_aug.append(m)
        else:
            row.append("—")
    best_err = min(errs_this_aug) if errs_this_aug else float('nan')
    best_per_aug[aug] = best_err
    row.append(f"{best_err:.4f}")
    table_data.append(row)
    row_colors.append(["#f9f9f9" if i % 2 == 0 else "white"] * len(col_labels))

for i, (aug, row_c) in enumerate(zip(exp4_augs_sorted, row_colors)):
    row_c[0] = "#e8e8f0"

# Find best aug at largest model
largest_wi = max(exp4_wis_sorted)
best_aug_largest = min(exp4_augs_sorted,
                       key=lambda a: exp4_summary.get(a, {}).get(largest_wi, (float('inf'),))[0])

tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.15, 1.7)

# Style header
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#2c3e50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Highlight best in each column
for j_idx, wi in enumerate(exp4_wis_sorted):
    col_j = j_idx + 1  # +1 for aug name col
    min_val = float('inf')
    min_row = -1
    for i, aug in enumerate(exp4_augs_sorted):
        if wi in exp4_summary.get(aug, {}):
            v = exp4_summary[aug][wi][0]
            if v < min_val:
                min_val = v
                min_row = i
    if min_row >= 0:
        tbl[min_row + 1, col_j].set_facecolor("#d4edda")
        tbl[min_row + 1, col_j].set_text_props(fontweight="bold")

# Highlight best "Best" column (last col index = len(col_labels) - 1)
best_best_aug = min(best_per_aug, key=best_per_aug.get)
last_col = len(col_labels) - 1
for i, aug in enumerate(exp4_augs_sorted):
    if aug == best_best_aug:
        tbl[i + 1, last_col].set_facecolor("#d4edda")
        tbl[i + 1, last_col].set_text_props(fontweight="bold")

ax.set_title("Table 1  CIFAR-100 Validation: ResNet Test Error by Augmentation Strategy\n"
             "(green = best in column)", pad=12, fontsize=11)

fig.savefig(os.path.join(FIGURES_DIR, "fig_table1_cifar100.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: fig_table1_cifar100.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Final summary printout for paper writing
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PAPER WRITING SUMMARY — KEY NUMBERS")
print("="*70)

print("\n[ResNet double descent peaks — CIFAR-10]")
for aug in AUG_ORDER:
    if aug not in peak_info:
        continue
    pi = peak_info[aug]
    print(f"  {AUG_LABELS[aug]:15s}: peak at w{pi['peak_wi']} "
          f"({params_to_M(pi['peak_params'])}), "
          f"peak_error={pi['peak_error']:.4f}")

print("\n[MixUp alpha ablation — best alpha]")
mean_errs = [(alpha, np.mean([exp3_summary[alpha][wi][0]
                               for wi in all_exp3_wi if wi in exp3_summary.get(alpha, {})]))
             for alpha in alphas]
best_alpha, best_alpha_err = min(mean_errs, key=lambda x: x[1])
print(f"  Best alpha overall: α={best_alpha:.1f} (mean_err={best_alpha_err:.4f})")
for alpha, err in sorted(mean_errs):
    print(f"  α={alpha:.1f}: mean_err={err:.4f}")

print("\n[CIFAR-100 — final (largest) model error per aug]")
for aug in exp4_augs_sorted:
    if aug not in exp4_summary:
        continue
    max_wi = max(exp4_summary[aug].keys())
    m, s, n = exp4_summary[aug][max_wi]
    print(f"  {AUG_LABELS.get(aug,aug):15s} (w{max_wi}, {params_to_M(RESNET_PARAMS.get(max_wi,0))}): "
          f"err={m:.4f}")

print("\n[CIFAR-100 — best (any model) error per aug]")
for aug in exp4_augs_sorted:
    print(f"  {AUG_LABELS.get(aug,aug):15s}: best={best_per_aug.get(aug, float('nan')):.4f}")

print("\n[MLP — best aug at each scale (largest 3 widths)]")
for wi in sorted(all_mlp_wi)[-3:]:
    best = min(AUG_ORDER, key=lambda a: mlp_summary.get(a, {}).get(wi, (float('inf'),))[0])
    err  = mlp_summary[best][wi][0]
    print(f"  wi={wi} ({params_to_M(MLP_PARAMS[wi])}): best aug={AUG_LABELS[best]}, err={err:.4f}")

print("\n[ResNet — w4/w5/w6 none-group detail (to confirm DD)]")
if "none" in resnet_summary:
    for wi in [4, 5, 6]:
        if wi in resnet_summary["none"]:
            m, s, n = resnet_summary["none"][wi]
            print(f"  w{wi} ({params_to_M(RESNET_PARAMS[wi])}): err={m:.4f} ± {s:.4f} (n={n})")

print("\nAll figures saved to:", FIGURES_DIR)
