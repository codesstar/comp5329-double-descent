"""
补跑小宽度 MLP（width=2,4,8），覆盖 double descent 驼峰区间。
结果存入同一个 results/metrics/ 目录，run_id 加 _extra 前缀区分。
用法：.venv/bin/python -m experiments.run_small_mlp
"""
import os, sys, json, traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import MLP
from src.augmentations import AUGMENT_NAMES
from src.data import get_loaders
from src.train import train_one_run

METRICS_DIR    = "results/metrics"
CHECKPOINT_DIR = "results/checkpoints"
SMALL_WIDTHS   = [2, 4, 8]  # width_idx = -3, -2, -1（负数避免和主实验冲突）
WIDTH_IDX_MAP  = {2: -3, 4: -2, 8: -1}

def build_runs():
    runs = []
    for aug in AUGMENT_NAMES:
        for w in SMALL_WIDTHS:
            wi = WIDTH_IDX_MAP[w]
            for seed in [0, 1, 2]:
                runs.append({
                    "run_id":    f"exp1_mlp_{aug}_wx{w}_s{seed}",
                    "exp":       "exp1",
                    "model":     "mlp",
                    "augment":   aug,
                    "width":     w,
                    "width_idx": wi,
                    "seed":      seed,
                    "dataset":   "cifar10",
                    "epochs":    200,
                })
    return runs

def load_completed():
    completed = set()
    for fname in os.listdir(METRICS_DIR):
        if not fname.endswith(".json"): continue
        try:
            d = json.load(open(os.path.join(METRICS_DIR, fname)))
            if d.get("status") == "completed":
                completed.add(d["run_id"])
        except Exception:
            pass
    return completed

def main():
    import torch
    all_runs  = build_runs()
    completed = load_completed()
    pending   = [r for r in all_runs if r["run_id"] not in completed]

    print(f"小宽度补跑：共 {len(all_runs)} runs，已完成 {len(all_runs)-len(pending)}，待跑 {len(pending)}")

    failed = []
    for i, cfg in enumerate(pending):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(pending)}] {cfg['run_id']}")
        print(f"  width={cfg['width']} params≈{MLP(cfg['width']).count_params():,}  aug={cfg['augment']}")
        print(f"{'='*60}")

        torch.manual_seed(cfg["seed"])
        try:
            train_loader, test_loader, n_classes = get_loaders(
                dataset="cifar10", augment=cfg["augment"], batch_size=128
            )
            model = MLP(width=cfg["width"], num_classes=n_classes)
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
            result.update({
                "width": cfg["width"], "width_idx": cfg["width_idx"],
                "augment": cfg["augment"], "model": "mlp",
                "exp": "exp1", "dataset": "cifar10",
            })
            with open(os.path.join(METRICS_DIR, f"{cfg['run_id']}.json"), "w") as f:
                json.dump(result, f, indent=2)
            print(f"  DONE: err={result['test_error']:.4f}  time={result['duration_min']:.1f}min")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed.append(cfg["run_id"])
            with open(os.path.join(METRICS_DIR, f"{cfg['run_id']}.json"), "w") as f:
                json.dump({"run_id": cfg["run_id"], "status": "failed", "error": str(e)}, f)

    print(f"\n补跑完成！成功 {len(pending)-len(failed)}  失败 {len(failed)}")
    if failed:
        for r in failed: print(f"  - {r}")

if __name__ == "__main__":
    main()
