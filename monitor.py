"""
实时进度监控：python monitor.py
"""
import os
import json
import sys
from datetime import datetime


METRICS_DIR   = "results/metrics"
PROGRESS_FILE = "results/progress.json"


def load_results():
    results = []
    if not os.path.exists(METRICS_DIR):
        return results
    for fname in os.listdir(METRICS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(METRICS_DIR, fname)) as f:
                results.append(json.load(f))
        except Exception:
            pass
    return results


def progress_bar(done, total, width=30):
    filled = int(width * done / max(total, 1))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {done}/{total} ({100*done/max(total,1):.1f}%)"


def main():
    results = load_results()
    completed = [r for r in results if r.get("status") == "completed"]
    failed    = [r for r in results if r.get("status") == "failed"]

    # 按实验分组
    exp_counts = {}
    exp_totals = {"exp1": 240, "exp3": 40, "exp4": 24}
    for r in completed:
        exp = r.get("exp", "unknown")
        exp_counts[exp] = exp_counts.get(exp, 0) + 1

    # 读进度文件
    in_progress = None
    total_runs = 380
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE) as f:
                prog = json.load(f)
            in_progress = prog.get("in_progress")
            total_runs  = prog.get("total_runs", 380)
            last_updated = prog.get("last_updated", "")
        except Exception:
            last_updated = ""
    else:
        last_updated = ""

    # 估算剩余时间（均值 15min/run）
    remaining = total_runs - len(completed)
    est_hours = remaining * 15 / 60

    print()
    print("=" * 55)
    print("  Deep Learning 实验进度监控")
    print("=" * 55)
    print(f"  总进度  {progress_bar(len(completed), total_runs)}")
    print(f"  失败    {len(failed)} runs")
    print(f"  预计剩余  {est_hours:.1f} 小时")
    if in_progress:
        print(f"  正在跑   {in_progress}")
    if last_updated:
        print(f"  更新时间  {last_updated[:19]}")
    print()
    print("  各实验进度:")
    for exp, total in sorted(exp_totals.items()):
        done = exp_counts.get(exp, 0)
        print(f"    {exp}  {progress_bar(done, total, width=20)}")
    print("=" * 55)

    if failed:
        print(f"\n  失败的 runs ({len(failed)}):")
        for r in failed[:10]:
            print(f"    - {r['run_id']}: {r.get('error','')[:60]}")
        if len(failed) > 10:
            print(f"    ... 还有 {len(failed)-10} 个")

    # 如果有结果，显示最佳结果预览
    if completed:
        print(f"\n  已完成 run 示例（最近5个）:")
        recent = sorted(completed, key=lambda r: r.get("run_id",""))[-5:]
        for r in recent:
            print(f"    {r['run_id']:<40} err={r.get('best_test_error',0):.4f}")
    print()


if __name__ == "__main__":
    main()
