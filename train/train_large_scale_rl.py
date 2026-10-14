"""
Large-Scale AlphaZero Self-Play Reinforcement Learning Pipeline for Omok++.
Features:
- Multi-Worker Parallel Self-Play (10 concurrent worker processes)
- 8-Fold Dihedral Symmetry Augmentation (D4 Group)
- High-Throughput GPU Batch Training (Batch Size 256 on RTX 4060 Ti)
- Continuous Arena Gating & Auto-Promotion
- Auto-Checkpointing after every iteration
"""

import os
import sys
import copy
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from collections import deque
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.board import Board
from models.policy_value_net import PolicyValueNet
from mcts.mcts import MCTS
from data.data_generator import augment_8fold


class LargeScaleReplayBuffer:
    def __init__(self, capacity: int = 150000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, policy: np.ndarray, value: float):
        self.buffer.append((state, policy, value))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)
        return (torch.from_numpy(np.array(states, dtype=np.float32)),
                torch.from_numpy(np.array(policies, dtype=np.float32)),
                torch.from_numpy(np.array(values, dtype=np.float32)).unsqueeze(1))

    def __len__(self):
        return len(self.buffer)


def worker_generate_games(worker_id: int,
                          model_weights: dict,
                          num_games: int,
                          mcts_sims: int,
                          out_queue: mp.Queue,
                          model_config: dict = None):
    """Worker process: loads model and plays num_games self-play games."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_config:
            model = PolicyValueNet.from_config(model_config).to(device)
        else:
            model = PolicyValueNet.create_baseline().to(device)
        model.load_state_dict(model_weights)
        model.eval()

        worker_samples = []

        for _ in range(num_games):
            board = Board()
            mcts = MCTS(model, device, c_puct=1.5, n_simulations=mcts_sims, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
            curr = Board.BLACK
            step = 0
            game_history = []

            while True:
                legal = board.get_legal_moves()
                if not legal:
                    winner = -1
                    break

                probs = mcts.search(board, curr, add_noise=True, n_sims=mcts_sims)
                if step < 6:
                    p_sum = np.sum(probs)
                    if p_sum > 0:
                        act = int(np.random.choice(361, p=probs / p_sum))
                    else:
                        act = int(np.argmax(probs))
                else:
                    act = int(np.argmax(probs))

                game_history.append((board.copy(), act, curr))
                board.place_stone(act // 19, act % 19, curr)
                mcts.step_root(act)
                step += 1

                over, winner = board.check_game_status()
                if over:
                    break
                curr = 1 - curr

            # 8-fold dihedral augmentation
            for b, act, p in game_history:
                st_tensor = PolicyValueNet.encode_board(b, p)
                if winner == -1 or winner is None:
                    val = 0.0
                elif winner == p:
                    val = 1.0
                else:
                    val = -1.0

                for aug_s, aug_a in augment_8fold(st_tensor, act):
                    worker_samples.append((aug_s, aug_a, val))

        out_queue.put((worker_id, num_games, worker_samples))
    except Exception as e:
        import traceback
        traceback.print_exc()
        out_queue.put((worker_id, 0, []))


def evaluate_arena_fast(cand_model: PolicyValueNet,
                        curr_model: PolicyValueNet,
                        device: torch.device,
                        num_games: int = 6,
                        sims: int = 40) -> float:
    cand_wins = 0
    curr_wins = 0
    draws = 0

    for g in range(num_games):
        board = Board()
        cand_is_black = (g % 2 == 0)
        cand_mcts = MCTS(cand_model, device, c_puct=1.5, n_simulations=sims)
        curr_mcts = MCTS(curr_model, device, c_puct=1.5, n_simulations=sims)
        active = Board.BLACK

        while True:
            legal = board.get_legal_moves()
            if not legal:
                draws += 1
                break

            if (active == Board.BLACK and cand_is_black) or (active == Board.WHITE and not cand_is_black):
                pi = cand_mcts.search(board, active, add_noise=False, n_sims=sims)
                act = int(np.argmax(pi))
            else:
                pi = curr_mcts.search(board, active, add_noise=False, n_sims=sims)
                act = int(np.argmax(pi))

            cand_mcts.step_root(act)
            curr_mcts.step_root(act)
            board.place_stone(act // 19, act % 19, active)

            over, winner = board.check_game_status()
            if over:
                if winner == -1:
                    draws += 1
                elif (winner == Board.BLACK and cand_is_black) or (winner == Board.WHITE and not cand_is_black):
                    cand_wins += 1
                else:
                    curr_wins += 1
                break
            active = 1 - active

    decisive = cand_wins + curr_wins
    return (cand_wins / max(1, decisive)) if decisive > 0 else 0.5


def train_large_scale_rl(total_games: int = 1000,
                         iters: int = 10,
                         workers: int = 10,
                         mcts_sims: int = 40,
                         epochs_per_iter: int = 3,
                         batch_size: int = 256,
                         lr: float = 5e-4,
                         model_type: str = "baseline"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(base_dir, "models", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    games_per_iter = total_games // iters
    games_per_worker = max(1, games_per_iter // workers)
    actual_games_per_iter = games_per_worker * workers

    print("=" * 70)
    print("      OMOK++ LARGE-SCALE ALPHAZERO REINFORCEMENT LEARNING      ")
    print("=" * 70)
    print(f"Total Target Games: {total_games} games across {iters} iterations")
    print(f"Per Iteration: {actual_games_per_iter} games ({workers} workers x {games_per_worker} games)")
    print(f"MCTS Simulations per Move: {mcts_sims}")
    print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Model Architecture: {model_type.upper()}")
    print(f"Batch Size: {batch_size} | Epochs per Iter: {epochs_per_iter} | Base LR: {lr}")
    print("-" * 70)

    # Initialize champion model
    if model_type == "2x":
        current_model = PolicyValueNet.create_2x().to(device)
    else:
        current_model = PolicyValueNet.create_baseline().to(device)

    base_ckpt = os.path.join(ckpt_dir, "rl_model_best.pt")
    if os.path.exists(base_ckpt):
        try:
            ckpt = torch.load(base_ckpt, map_location=device)
            st = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            current_model.load_state_dict(st)
            print(f"[INIT] Loaded champion weights from {base_ckpt}")
        except Exception as e:
            print(f"[INIT] Checkpoint shape mismatch or fresh 2x model requested ({e}). Initializing clean weights.")
    else:
        print("[INIT] Starting fresh champion weights.")

    replay_buffer = LargeScaleReplayBuffer(capacity=250000)

    # Preload stress test samples if available
    stress_file = os.path.join(base_dir, "data", "stress_test_samples.npz")
    if os.path.exists(stress_file):
        try:
            d = np.load(stress_file)
            st, ac, va = d["states"], d["actions"], d["values"]
            for s, a, v in zip(st, ac, va):
                pi = np.zeros(361, dtype=np.float32)
                pi[a] = 1.0
                replay_buffer.push(s, pi, float(v))
            print(f"[REPLAY BUFFER] Preloaded {len(st):,} samples from stress test data!")
        except Exception as e:
            print(f"[REPLAY BUFFER] Could not preload stress samples: {e}")

    incoming_dir = os.path.join(base_dir, "data", "incoming")
    os.makedirs(incoming_dir, exist_ok=True)

    start_iter = 1
    if os.path.exists(base_ckpt):
        try:
            ckpt = torch.load(base_ckpt, map_location="cpu")
            if isinstance(ckpt, dict) and "iteration" in ckpt:
                start_iter = ckpt["iteration"] + 1
                print(f"[RESUME] Resuming from Iteration {start_iter} (Champion was Iteration {ckpt['iteration']})")
        except Exception:
            pass

    criterion_p = nn.CrossEntropyLoss()
    criterion_v = nn.MSELoss()
    logs = []
    overall_start = time.time()

    end_iter = start_iter + iters - 1
    for it in range(start_iter, end_iter + 1):
        it_start = time.time()
        print(f"\n>>> ITERATION {it}/{end_iter} (Target: {actual_games_per_iter} games) <<<", flush=True)

        # Ingest any external worker samples from incoming directory
        ext_samples = 0
        if os.path.exists(incoming_dir):
            for fname in os.listdir(incoming_dir):
                if fname.endswith(".npz"):
                    fpath = os.path.join(incoming_dir, fname)
                    try:
                        d = np.load(fpath)
                        st, ac, va = d["states"], d["actions"], d["values"]
                        for s, a, v in zip(st, ac, va):
                            pi = np.zeros(361, dtype=np.float32)
                            pi[a] = 1.0
                            replay_buffer.push(s, pi, float(v))
                        ext_samples += len(st)
                        os.remove(fpath)
                    except Exception as e:
                        print(f"  [EXTERNAL] Error reading {fname}: {e}")
        if ext_samples > 0:
            print(f"  [EXTERNAL WORKERS] Ingested {ext_samples:,} samples from external worker nodes! (Buffer: {len(replay_buffer):,})", flush=True)

        # 1. Parallel Self-Play
        t0 = time.time()
        out_queue = mp.Queue()
        weights_cpu = {k: v.cpu() for k, v in current_model.state_dict().items()}
        procs = []

        for w_id in range(workers):
            p = mp.Process(target=worker_generate_games,
                           args=(w_id, weights_cpu, games_per_worker, mcts_sims, out_queue, current_model.get_config()))
            p.daemon = True
            p.start()
            procs.append(p)

        new_samples = 0
        games_collected = 0
        for _ in range(workers):
            w_id, g_cnt, s_list = out_queue.get()
            games_collected += g_cnt
            new_samples += len(s_list)
            for s, a, v in s_list:
                pi = np.zeros(361, dtype=np.float32)
                pi[a] = 1.0
                replay_buffer.push(s, pi, v)

        for p in procs:
            p.join(timeout=3.0)

        gen_time = time.time() - t0
        print(f"  [Self-Play] Generated {games_collected} games (+{new_samples:,} samples) in {gen_time:.1f}s ({games_collected/gen_time:.2f} games/s)", flush=True)
        print(f"  [Replay Buffer] Total Samples in Buffer: {len(replay_buffer):,}", flush=True)

        # 2. Train Candidate Model
        candidate_model = copy.deepcopy(current_model).to(device)
        candidate_model.train()
        optimizer = optim.AdamW(candidate_model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_iter)

        num_batches = max(10, min(200, len(replay_buffer) // batch_size))
        print(f"  [Training] Training candidate on {num_batches} batches/epoch x {epochs_per_iter} epochs (Batch: {batch_size})...", flush=True)

        t_tr = time.time()
        ep_loss_p = 0.0
        ep_loss_v = 0.0

        for ep in range(epochs_per_iter):
            loss_p_sum = 0.0
            loss_v_sum = 0.0
            for _ in range(num_batches):
                b_states, b_policies, b_values = replay_buffer.sample(batch_size)
                b_states = b_states.to(device)
                b_policies = b_policies.to(device)
                b_values = b_values.to(device)

                optimizer.zero_grad()
                p_logits, v_pred = candidate_model(b_states)
                targets = torch.argmax(b_policies, dim=1)

                lp = criterion_p(p_logits, targets)
                lv = criterion_v(v_pred, b_values)
                loss = lp + 0.5 * lv
                loss.backward()
                nn.utils.clip_grad_norm_(candidate_model.parameters(), 2.0)
                optimizer.step()

                loss_p_sum += lp.item()
                loss_v_sum += lv.item()

            scheduler.step()
            ep_loss_p = loss_p_sum / num_batches
            ep_loss_v = loss_v_sum / num_batches

        train_time = time.time() - t_tr
        print(f"  [Training] Completed in {train_time:.1f}s | Policy Loss: {ep_loss_p:.4f} | Value Loss: {ep_loss_v:.4f}", flush=True)

        # 3. Arena Gating
        candidate_model.eval()
        current_model.eval()
        win_rate = evaluate_arena_fast(candidate_model, current_model, device, num_games=6, sims=30)
        promoted = win_rate >= 0.50
        print(f"  [Arena] Candidate Win Rate vs Champion: {win_rate*100:.1f}% -> {'PROMOTED!' if promoted else 'Rejected (Keep Champion)'}", flush=True)

        if promoted:
            current_model = candidate_model
            torch.save({
                "iteration": it,
                "model_state_dict": current_model.state_dict(),
                "config": current_model.get_config()
            }, os.path.join(ckpt_dir, "rl_model_best.pt"))

        # Save iteration checkpoint
        iter_ckpt = os.path.join(ckpt_dir, f"rl_model_iter_{it}.pt")
        torch.save({
            "iteration": it,
            "model_state_dict": current_model.state_dict(),
            "config": current_model.get_config(),
            "win_rate": win_rate,
            "promoted": promoted
        }, iter_ckpt)

        it_elapsed = time.time() - it_start
        cum_elapsed = time.time() - overall_start
        print(f"  [Iteration {it}] Done in {it_elapsed:.1f}s (Total Elapsed: {cum_elapsed/60:.1f}m)", flush=True)

        logs.append({
            "iteration": it,
            "games": actual_games_per_iter * it,
            "samples": len(replay_buffer),
            "policy_loss": ep_loss_p,
            "value_loss": ep_loss_v,
            "win_rate": win_rate,
            "promoted": promoted,
            "time_sec": it_elapsed
        })

    with open(os.path.join(ckpt_dir, "rl_large_scale_training_log.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  LARGE-SCALE SELF-PLAY RL COMPLETE! (Total Elapsed: {(time.time()-overall_start)/60:.1f}m)")
    print(f"  Best Champion Checkpoint Saved to: {os.path.join(ckpt_dir, 'rl_model_best.pt')}")
    print("=" * 70)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_games", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--sims", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "2x"])
    args = parser.parse_args()

    train_large_scale_rl(total_games=args.total_games,
                         iters=args.iters,
                         workers=args.workers,
                         mcts_sims=args.sims,
                         epochs_per_iter=args.epochs,
                         batch_size=args.batch,
                         model_type=args.model_type)
