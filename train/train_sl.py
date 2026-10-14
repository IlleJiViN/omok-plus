"""
Supervised Learning Pipeline for PolicyValueNet.
Pretrains Policy and Value heads using synthetic Omok++ self-play data.
Tracks:
- Policy Top-1 and Top-5 accuracy
- Value MSE loss and direction accuracy
- Training and validation loss curves
- Saves best model checkpoints and metadata
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.policy_value_net import PolicyValueNet


class OmokDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.states = torch.from_numpy(data['states']).float()       # (N, 7, 19, 19)
        self.policies = torch.from_numpy(data['policies']).float()   # (N, 361)
        self.values = torch.from_numpy(data['values']).float().unsqueeze(1) # (N, 1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


def evaluate_metrics(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_samples = 0
    total_p_loss = 0.0
    total_v_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    sign_correct = 0

    criterion_p = nn.CrossEntropyLoss()
    criterion_v = nn.MSELoss()

    with torch.no_grad():
        for states, policies, values in loader:
            states, policies, values = states.to(device), policies.to(device), values.to(device)
            p_logits, v_pred = model(states)

            # Targets
            target_indices = torch.argmax(policies, dim=1)
            p_loss = criterion_p(p_logits, target_indices)
            v_loss = criterion_v(v_pred, values)

            batch_size = states.size(0)
            total_samples += batch_size
            total_p_loss += p_loss.item() * batch_size
            total_v_loss += v_loss.item() * batch_size

            # Top-1 accuracy
            preds = torch.argmax(p_logits, dim=1)
            top1_correct += (preds == target_indices).sum().item()

            # Top-5 accuracy
            _, top5_preds = torch.topk(p_logits, k=5, dim=1)
            top5_correct += (top5_preds == target_indices.unsqueeze(1)).any(dim=1).sum().item()

            # Value sign accuracy (excluding 0)
            non_zero_mask = (values.abs() > 0.01)
            if non_zero_mask.any():
                correct_signs = ((v_pred * values) > 0)[non_zero_mask]
                sign_correct += correct_signs.sum().item()

    avg_p_loss = total_p_loss / max(1, total_samples)
    avg_v_loss = total_v_loss / max(1, total_samples)
    top1_acc = (top1_correct / max(1, total_samples)) * 100.0
    top5_acc = (top5_correct / max(1, total_samples)) * 100.0
    sign_acc = (sign_correct / max(1, total_samples)) * 100.0

    return {
        "p_loss": avg_p_loss,
        "v_loss": avg_v_loss,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "sign_acc": sign_acc
    }


def train_sl(train_path: str, val_path: str, output_dir: str, epochs: int = 15, batch_size: int = 64, lr: float = 1e-3, num_channels: int = 64, num_res_blocks: int = 5):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (cuda={torch.cuda.is_available()})")

    print(f"Loading training data from {train_path}...")
    train_ds = OmokDataset(train_path)
    val_ds = OmokDataset(val_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    model = PolicyValueNet(in_channels=7, num_channels=num_channels, num_res_blocks=num_res_blocks).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion_p = nn.CrossEntropyLoss()
    criterion_v = nn.MSELoss()

    best_val_loss = float('inf')
    history = []

    print("\n--- Starting Supervised Pretraining ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_p_loss_sum = 0.0
        train_v_loss_sum = 0.0
        train_samples = 0

        for states, policies, values in train_loader:
            states, policies, values = states.to(device), policies.to(device), values.to(device)
            optimizer.zero_grad()

            p_logits, v_pred = model(states)
            target_indices = torch.argmax(policies, dim=1)

            loss_p = criterion_p(p_logits, target_indices)
            loss_v = criterion_v(v_pred, values)
            loss = loss_p + 0.5 * loss_v

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            bs = states.size(0)
            train_p_loss_sum += loss_p.item() * bs
            train_v_loss_sum += loss_v.item() * bs
            train_samples += bs

        scheduler.step()

        # Validation evaluation
        val_metrics = evaluate_metrics(model, val_loader, device)
        train_avg_p = train_p_loss_sum / max(1, train_samples)
        train_avg_v = train_v_loss_sum / max(1, train_samples)
        total_val_loss = val_metrics["p_loss"] + 0.5 * val_metrics["v_loss"]

        epoch_record = {
            "epoch": epoch,
            "train_p_loss": train_avg_p,
            "train_v_loss": train_avg_v,
            "val_p_loss": val_metrics["p_loss"],
            "val_v_loss": val_metrics["v_loss"],
            "val_top1_acc": val_metrics["top1_acc"],
            "val_top5_acc": val_metrics["top5_acc"],
            "val_sign_acc": val_metrics["sign_acc"],
            "lr": optimizer.param_groups[0]["lr"]
        }
        history.append(epoch_record)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train P: {train_avg_p:.3f} V: {train_avg_v:.3f} | "
              f"Val P: {val_metrics['p_loss']:.3f} V: {val_metrics['v_loss']:.3f} | "
              f"Top-1: {val_metrics['top1_acc']:.2f}% Top-5: {val_metrics['top5_acc']:.2f}% | "
              f"ValLoss: {total_val_loss:.3f}")

        # Save checkpoint if best
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_path = os.path.join(output_dir, "sl_model_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": total_val_loss,
                "metrics": val_metrics,
                "config": {
                    "in_channels": 7,
                    "num_channels": num_channels,
                    "num_res_blocks": num_res_blocks
                }
            }, best_path)

    # Save latest and history
    latest_path = os.path.join(output_dir, "sl_model_latest.pt")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "config": {
            "in_channels": 7,
            "num_channels": num_channels,
            "num_res_blocks": num_res_blocks
        }
    }, latest_path)

    log_path = os.path.join(output_dir, "sl_training_history.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s. Best Val Loss: {best_val_loss:.4f}")
    print(f"Best checkpoint saved to: {os.path.join(output_dir, 'sl_model_best.pt')}")
    print(f"History log saved to: {log_path}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/dataset/train_data.npz")
    parser.add_argument("--val", type=str, default="data/dataset/val_data.npz")
    parser.add_argument("--out", type=str, default="models/checkpoints")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=5)
    args = parser.parse_args()

    train_sl(args.train, args.val, args.out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, num_channels=args.channels, num_res_blocks=args.blocks)
