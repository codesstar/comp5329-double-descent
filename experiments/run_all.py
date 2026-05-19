"""
主实验调度器：自动跳过已完成的 run，支持断点续跑。
用法：.venv/bin/python -m experiments.run_all
"""
import os
import sys
import json
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import MLP, MLP_WIDTHS, ResNet, RESNET_WIDTHS
from src.augmentations import AUGMENT_NAMES
from src.data import get_loaders
from src.train import train_one_run

METRICS_DIR    = "results/metrics"
PROGRESS_FILE  = "results/progress.json"
CHECKPOINT_DIR = "results/checkpoints"

# ── 实验配置 ──────────────────────────────────────────────────────────

def build_run_list():
    runs = []

    # Exp1: MLP × 5增强 × 8宽度 × 3种子
    for aug in AUGMENT_NAMES:
        for wi, w in enumerate(MLP_WIDTHS):
            for seed in [0, 1, 2]:
                runs.append({
                    "run_id":  f"exp1_mlp_{aug}_w{wi}_s{seed}",
                    "exp":     "exp1",
                    "model":   "mlp",
                    "augment": aug,
                    "width":   w,
                    "width_idx": wi,
                    "seed":    seed,
                    "dataset": "cifar10",
                    "epochs":  200,
                })

    # Exp1: ResNet × 5增强 × 7宽度（去掉w7最慢档） × 3种子
    for aug in AUGMENT_NAMES:
        for wi, w in enumerate(RESNET_WIDTHS):
            if wi == 7:
                continue  # w7每run约3小时，时间不够，跳过
            for seed in [0, 1, 2]:
                runs.append({
                    "run_id":  f"exp1_resnet_{aug}_w{wi}_s{seed}",
                    "exp":     "exp1",
                    "model":   "resnet",
                    "augment": aug,
                    "width":   w,
                    "width_idx": wi,
                    "seed":    seed,
                    "dataset": "cifar10",
                    "epochs":  200,
                })

    # Exp3: MixUp 强度 ablation（5强度 × 8宽度 × 1种子）
    for alpha in [0.1, 0.2, 0.4, 0.8, 1.0]:
        alpha_str = str(alpha).replace(".", "p")
        for wi, w in enumerate(MLP_WIDTHS):
            runs.append({
                "run_id":   f"exp3_mixup_a{alpha_str}_w{wi}_s0",
                "exp":      "exp3",
                "model":    "mlp",
                "augment":  "mixup",
                "mixup_alpha": alpha,
                "width":    w,
                "width_idx": wi,
                "seed":     0,
                "dataset":  "cifar10",
                "epochs":   200,
            })

    # Exp4: ResNet × cifar100 × 3增强 × 7宽度（去掉w7） × 1种子
    for aug in ["none", "mixup", "cutout"]:
        for wi, w in enumerate(RESNET_WIDTHS):
            if wi == 7:
                continue  # 同上，跳过w7
            runs.append({
                "run_id":  f"exp4_resnet_{aug}_w{wi}_s0",
                "exp":     "exp4",
                "model":   "resnet",
                "augment": aug,
                "width":   w,
                "width_idx": wi,
                "seed":    0,
                "dataset": "cifar100",
                "epochs":  200,
            })

    return runs


# ── 进度管理 ───────────────────────────────────────────────────────────

def load_completed() -> set:
    os.makedirs(METRICS_DIR, exist_ok=True)
    completed = set()
    for fname in os.listdir(METRICS_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(METRICS_DIR, fname)
        try:
            with open(path) as f:
                d = json.load(f)
            if d.get("status") == "completed":
                completed.add(d["run_id"])
        except Exception:
            pass
    return completed


def save_progress(total, completed_n, current_run, failed_runs):
    os.makedirs("results", exist_ok=True)
    data = {
        "total_runs": total,
        "completed": completed_n,
        "failed": len(failed_runs),
        "in_progress": current_run,
        "failed_runs": failed_runs,
        "last_updated": datetime.now().isoformat(),
    }
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── 单次 run 执行 ──────────────────────────────────────────────────────

def execute_run(cfg: dict) -> bool:
    import torch
    torch.manual_seed(cfg["seed"])

    dataset = cfg["dataset"]
    n_classes = 10 if dataset == "cifar10" else 100
    train_loader, test_loader, n_classes = get_loaders(
        dataset=dataset,
        augment=cfg["augment"],
        batch_size=128,
    )

    if cfg["model"] == "mlp":
        model = MLP(width=cfg["width"], num_classes=n_classes)
    else:
        model = ResNet(base_width=cfg["width"], num_classes=n_classes)

    n_params = model.count_params()
    print(f"\n{'='*60}")
    print(f"RUN: {cfg['run_id']}")
    print(f"  model={cfg['model']} width={cfg['width']} params={n_params:,}")
    print(f"  augment={cfg['augment']} dataset={dataset} seed={cfg['seed']}")
    print(f"{'='*60}")

    result = train_one_run(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        run_id=cfg["run_id"],
        augment=cfg["augment"],
        n_classes=n_classes,
        epochs=cfg["epochs"],
        checkpoint_dir=CHECKPOINT_DIR,
        metrics_dir=METRICS_DIR,
    )
    result["width"] = cfg["width"]
    result["width_idx"] = cfg["width_idx"]
    result["augment"] = cfg["augment"]
    result["model"] = cfg["model"]
    result["exp"] = cfg["exp"]
    result["dataset"] = dataset

    with open(os.path.join(METRICS_DIR, f"{cfg['run_id']}.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"  DONE: test_error={result['test_error']:.4f} "
          f"best={result['best_test_error']:.4f} "
          f"time={result['duration_min']:.1f}min")
    return True


# ── 主入口 ────────────────────────────────────────────────────────────

def main():
    all_runs = build_run_list()
    total = len(all_runs)
    print(f"总计 {total} 个 runs")

    completed = load_completed()
    pending = [r for r in all_runs if r["run_id"] not in completed]
    print(f"已完成: {len(completed)}  待运行: {len(pending)}")

    if not pending:
        print("所有实验已完成！")
        return

    failed_runs = []

    for i, cfg in enumerate(pending):
        save_progress(total, len(completed), cfg["run_id"], failed_runs)

        try:
            execute_run(cfg)
            completed.add(cfg["run_id"])
        except Exception as e:
            print(f"\n[FAILED] {cfg['run_id']}: {e}")
            traceback.print_exc()
            failed_runs.append(cfg["run_id"])
            # 记录失败状态
            fail_path = os.path.join(METRICS_DIR, f"{cfg['run_id']}.json")
            with open(fail_path, "w") as f:
                json.dump({
                    "run_id": cfg["run_id"],
                    "status": "failed",
                    "error": str(e),
                }, f)

        pct = len(completed) / total * 100
        print(f"\n进度: {len(completed)}/{total} ({pct:.1f}%) | 失败: {len(failed_runs)}")

    save_progress(total, len(completed), None, failed_runs)
    print(f"\n{'='*60}")
    print(f"全部完成！成功: {len(completed)-len(failed_runs)}  失败: {len(failed_runs)}")
    if failed_runs:
        print("失败的 runs:")
        for r in failed_runs:
            print(f"  - {r}")


if __name__ == "__main__":
    main()
