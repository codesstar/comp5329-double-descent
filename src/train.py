import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any

from .augmentations import mixup_batch


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_run(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    run_id: str,
    augment: str,
    n_classes: int,
    epochs: int = 200,
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    checkpoint_dir: str = "results/checkpoints",
    metrics_dir: str = "results/metrics",
) -> Dict[str, Any]:

    device = get_device()
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr,
        momentum=0.9, weight_decay=weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    ckpt_path   = os.path.join(checkpoint_dir, f"{run_id}_latest.pt")
    best_path   = os.path.join(checkpoint_dir, f"{run_id}_best.pt")
    metric_path = os.path.join(metrics_dir, f"{run_id}.json")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # ── 从 checkpoint 恢复 ──────────────────────────────────────────────
    start_epoch = 0
    best_test_error = 1.0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_test_error = ckpt.get("best_test_error", 1.0)
        print(f"  [resume] {run_id} from epoch {start_epoch}")

    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        # ── 训练 ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            if augment == "mixup":
                x, y_soft = mixup_batch(x, y, alpha=0.4, num_classes=n_classes)
                logits = model(x)
                loss = -(y_soft * torch.log_softmax(logits, dim=1)).sum(1).mean()
            else:
                logits = model(x)
                loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # ── 每 50 epoch 评估 + 保存 checkpoint ────────────────────────
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            test_error = evaluate(model, test_loader, device)
            if test_error < best_test_error:
                best_test_error = test_error
                torch.save(model.state_dict(), best_path)

            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_test_error": best_test_error,
            }, ckpt_path)

            elapsed = (time.time() - t0) / 60
            print(f"  epoch {epoch+1:3d}/{epochs} | loss {train_loss:.4f} | "
                  f"err {test_error:.4f} | {elapsed:.1f}min")

    # ── 最终评估 ──────────────────────────────────────────────────────
    final_error = evaluate(model, test_loader, device)
    duration_min = (time.time() - t0) / 60

    result = {
        "run_id": run_id,
        "status": "completed",
        "n_params": sum(p.numel() for p in model.parameters()),
        "train_loss": round(train_loss, 6),
        "test_error": round(final_error, 6),
        "best_test_error": round(best_test_error, 6),
        "duration_min": round(duration_min, 2),
    }
    with open(metric_path, "w") as f:
        json.dump(result, f, indent=2)

    # 清理 latest checkpoint（保留 best）
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return result


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 1.0 - correct / total
