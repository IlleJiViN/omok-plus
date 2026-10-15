"""
Fine-tune ResNet-10 Policy-Value Network with Expert Distillation Data.
Refines policy prior to place high probabilities on tactical fork/threat blocks,
eliminating tactical blindspots purely within the neural network weights.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.policy_value_net import PolicyValueNet


def finetune_neural_model(
    data_path: str = "data/distilled/teacher_data.npz",
    base_ckpt_path: str = "models/checkpoints/rl_model_best.pt",
    out_ckpt_path: str = "models/checkpoints/rl_model_pure_neural.pt",
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-4,
    device_name: str = "cuda"
):
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"Loading distilled data from {data_path}...")
    data = np.load(data_path)

    train_states = torch.from_numpy(data['train_states']).float()
    train_moves = torch.from_numpy(data['train_moves']).long()
    train_values = torch.from_numpy(data['train_values']).float().unsqueeze(1)

    val_states = torch.from_numpy(data['val_states']).float()
    val_moves = torch.from_numpy(data['val_moves']).long()
    val_values = torch.from_numpy(data['val_values']).float().unsqueeze(1)

    print(f"Train samples: {len(train_states):,}, Val samples: {len(val_states):,}")

    train_loader = DataLoader(TensorDataset(train_states, train_moves, train_values), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_states, val_moves, val_values), batch_size=batch_size, shuffle=False)

    print(f"Loading champion model from {base_ckpt_path} on {device}...")
    ckpt = torch.load(base_ckpt_path, map_location=device)
    config = ckpt.get("config") or ckpt.get("model_config")
    st = ckpt.get("model_state_dict") or ckpt

    if config:
        model = PolicyValueNet.from_config(config).to(device)
    else:
        model = PolicyValueNet.create_2x().to(device)
    model.load_state_dict(st)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion_p = nn.CrossEntropyLoss()
    criterion_v = nn.MSELoss()

    best_val_acc = 0.0
    start_t = time.time()

    print("\nStarting Knowledge Distillation Fine-Tuning:")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_p_loss, total_v_loss = 0.0, 0.0, 0.0
        correct_p, total_samples = 0, 0

        for b_states, b_moves, b_values in train_loader:
            b_states = b_states.to(device)
            b_moves = b_moves.to(device)
            b_values = b_values.to(device)

            optimizer.zero_grad()
            p_logits, v_pred = model(b_states)

            loss_p = criterion_p(p_logits, b_moves)
            loss_v = criterion_v(v_pred, b_values)
            loss = loss_p + 0.5 * loss_v

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(b_states)
            total_p_loss += loss_p.item() * len(b_states)
            total_v_loss += loss_v.item() * len(b_states)

            pred_moves = p_logits.argmax(dim=-1)
            correct_p += (pred_moves == b_moves).sum().item()
            total_samples += len(b_states)

        scheduler.step()

        train_acc = (correct_p / total_samples) * 100.0
        avg_loss = total_loss / total_samples

        # Validation
        model.eval()
        val_loss, val_p_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for b_states, b_moves, b_values in val_loader:
                b_states = b_states.to(device)
                b_moves = b_moves.to(device)
                b_values = b_values.to(device)

                p_logits, v_pred = model(b_states)
                loss_p = criterion_p(p_logits, b_moves)
                loss_v = criterion_v(v_pred, b_values)
                loss = loss_p + 0.5 * loss_v

                val_loss += loss.item() * len(b_states)
                val_p_correct += (p_logits.argmax(dim=-1) == b_moves).sum().item()
                val_total += len(b_states)

        val_acc = (val_p_correct / val_total) * 100.0
        avg_val_loss = val_loss / val_total

        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_loss:.4f} (Acc: {train_acc:.2f}%) | Val Loss: {avg_val_loss:.4f} (Val Acc: {val_acc:.2f}%) | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_val_acc or epoch == epochs:
            best_val_acc = max(best_val_acc, val_acc)
            save_dict = {
                "model_state_dict": model.state_dict(),
                "config": model.get_config(),
                "val_acc": val_acc,
                "epoch": epoch,
                "fine_tuned_on": "friend_agent_expert_distill"
            }
            torch.save(save_dict, out_ckpt_path)

    print(f"\n[DONE] Distilled Model saved to {out_ckpt_path} (Best Val Policy Acc: {best_val_acc:.2f}%) in {time.time() - start_t:.1f}s")


if __name__ == "__main__":
    finetune_neural_model()
