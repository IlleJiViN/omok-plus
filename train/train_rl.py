"""
AlphaZero-style Self-Play Reinforcement Learning Pipeline for Omok++.
Iterative loop:
1. Self-play with MCTS + Dirichlet noise
2. Replay buffer data collection with 8-fold augmentation
3. Policy-Value Network training (CrossEntropy + MSE)
4. Arena evaluation against previous checkpoint (win-rate gating)
5. Save iteration checkpoints and progress logs
"""

import os
import sys
import copy
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.board import Board
from engine.gamemanager import GameManager
from models.policy_value_net import PolicyValueNet
from mcts.mcts import MCTS
from data.data_generator import augment_8fold


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
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


def self_play_game(model: PolicyValueNet, device: torch.device, n_sims: int = 60, temp_threshold: int = 8) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Executes a single self-play game using MCTS.
    Returns list of (state_tensor, policy_361, value_target).
    """
    board = Board()
    mcts = MCTS(model, device, c_puct=1.5, n_simulations=n_sims, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
    game_history: List[Tuple[np.ndarray, np.ndarray, int]] = [] # (state, pi, player)

    curr_player = Board.BLACK
    step = 0

    while True:
        legal = board.get_legal_moves()
        if not legal:
            winner = -1
            break

        state_tensor = PolicyValueNet.encode_board(board, curr_player)
        action_probs = mcts.search(board, curr_player, add_noise=True, n_sims=n_sims)

        game_history.append((state_tensor, action_probs, curr_player))

        # Temperature-based action selection
        if step < temp_threshold:
            # Temperature tau = 1.0 -> sample proportional to visit counts
            # Add small epsilon to avoid numerical issues
            probs = action_probs.copy()
            probs_sum = np.sum(probs)
            if probs_sum > 0:
                probs /= probs_sum
                action = int(np.random.choice(361, p=probs))
            else:
                action = int(np.argmax(probs))
        else:
            # Temperature tau -> 0: greedy selection
            action = int(np.argmax(action_probs))

        r, c = divmod(action, 19)
        board.place_stone(r, c, curr_player)
        mcts.step_root(action)
        step += 1

        is_over, winner = board.check_game_status()
        if is_over:
            break

        curr_player = 1 - curr_player

    # Build targets with 8-fold augmentation
    samples = []
    for state_tensor, policy_probs, player in game_history:
        if winner == -1 or winner is None:
            val = 0.0
        elif winner == player:
            val = 1.0
        else:
            val = -1.0

        # Augmentation
        best_act = int(np.argmax(policy_probs))
        augmented = augment_8fold(state_tensor, best_act)
        for aug_state, aug_act in augmented:
            aug_pi = np.zeros(361, dtype=np.float32)
            aug_pi[aug_act] = 1.0
            samples.append((aug_state, aug_pi, val))

    return samples


def evaluate_arena(cand_model: PolicyValueNet,
                   curr_model: PolicyValueNet,
                   device: torch.device,
                   num_games: int = 8,
                   sims: int = 50) -> float:
    """
    Plays candidate model against current model in head-to-head match.
    Alternates colors.
    Returns candidate win rate.
    """
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
                # Candidate turn
                pi = cand_mcts.search(board, active, add_noise=False, n_sims=sims)
                act = int(np.argmax(pi))
                cand_mcts.step_root(act)
                curr_mcts.step_root(act)
            else:
                # Current model turn
                pi = curr_mcts.search(board, active, add_noise=False, n_sims=sims)
                act = int(np.argmax(pi))
                cand_mcts.step_root(act)
                curr_mcts.step_root(act)

            r, c = divmod(act, 19)
            board.place_stone(r, c, active)

            is_over, winner = board.check_game_status()
            if is_over:
                if winner == -1:
                    draws += 1
                elif (winner == Board.BLACK and cand_is_black) or (winner == Board.WHITE and not cand_is_black):
                    cand_wins += 1
                else:
                    curr_wins += 1
                break

            active = 1 - active

    total_decisive = cand_wins + curr_wins
    cand_win_rate = (cand_wins / max(1, total_decisive)) if total_decisive > 0 else 0.5
    print(f"  [Arena] Cand Wins: {cand_wins}, Prev Wins: {curr_wins}, Draws: {draws} (Win Rate: {cand_win_rate*100:.1f}%)", flush=True)
    return cand_win_rate


def run_self_play_rl(base_model_path: str,
                     output_dir: str,
                     iterations: int = 3,
                     games_per_iter: int = 15,
                     mcts_sims: int = 50,
                     train_epochs: int = 3,
                     batch_size: int = 64,
                     lr: float = 5e-4):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"RL Training on: {device}")

    # Load baseline model
    current_model = PolicyValueNet(in_channels=7, num_channels=64, num_res_blocks=5).to(device)
    if os.path.exists(base_model_path):
        ckpt = torch.load(base_model_path, map_location=device)
        st = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        current_model.load_state_dict(st)
        print(f"Loaded base model from {base_model_path}")
    else:
        print("Starting from scratch (no base checkpoint found).")

    replay_buffer = ReplayBuffer(capacity=40000)
    optimizer = optim.AdamW(current_model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_p = nn.CrossEntropyLoss()
    criterion_v = nn.MSELoss()

    rl_logs = []

    for it in range(1, iterations + 1):
        print(f"\n=================== RL Iteration {it}/{iterations} ===================")
        iter_start = time.time()

        # 1. Self-Play
        print(f"Generating {games_per_iter} self-play games (sims={mcts_sims})...", flush=True)
        current_model.eval()
        new_samples = 0
        for g in range(games_per_iter):
            g_start = time.time()
            samples = self_play_game(current_model, device, n_sims=mcts_sims)
            for s, p, v in samples:
                replay_buffer.push(s, p, v)
            new_samples += len(samples)
            print(f"  Game {g+1}/{games_per_iter} complete (+{len(samples)} samples, {time.time()-g_start:.1f}s)", flush=True)

        print(f"Buffer size: {len(replay_buffer)} (+{new_samples} new augmented samples)", flush=True)

        # 2. Train candidate model
        candidate_model = copy.deepcopy(current_model).to(device)
        candidate_optimizer = optim.AdamW(candidate_model.parameters(), lr=lr, weight_decay=1e-4)

        candidate_model.train()
        num_batches = max(10, len(replay_buffer) // batch_size)
        print(f"Training candidate model for {train_epochs} epochs ({num_batches} batches/epoch)...")

        for ep in range(train_epochs):
            loss_p_total = 0.0
            loss_v_total = 0.0
            for _ in range(num_batches):
                b_states, b_policies, b_values = replay_buffer.sample(batch_size)
                b_states = b_states.to(device)
                b_policies = b_policies.to(device)
                b_values = b_values.to(device)

                candidate_optimizer.zero_grad()
                p_logits, v_pred = candidate_model(b_states)
                targets = torch.argmax(b_policies, dim=1)

                loss_p = criterion_p(p_logits, targets)
                loss_v = criterion_v(v_pred, b_values)
                loss = loss_p + 0.5 * loss_v

                loss.backward()
                nn.utils.clip_grad_norm_(candidate_model.parameters(), 2.0)
                candidate_optimizer.step()

                loss_p_total += loss_p.item()
                loss_v_total += loss_v.item()

            print(f"  Epoch {ep+1}: Loss P: {loss_p_total/num_batches:.3f}, Loss V: {loss_v_total/num_batches:.3f}")

        # 3. Arena Evaluation (Gating)
        print("Evaluating Candidate vs Current in Arena...")
        candidate_model.eval()
        current_model.eval()
        win_rate = evaluate_arena(candidate_model, current_model, device, num_games=6, sims=40)

        # Update if win rate > 50%
        promoted = False
        if win_rate >= 0.50:
            print(f">>> CANDIDATE PROMOTED to new champion! (Win rate: {win_rate*100:.1f}%)")
            current_model = candidate_model
            promoted = True
        else:
            print(f">>> Candidate rejected. Keeping current model.")

        iter_elapsed = time.time() - iter_start

        # Save checkpoint
        iter_ckpt_path = os.path.join(output_dir, f"rl_model_iter_{it}.pt")
        best_ckpt_path = os.path.join(output_dir, "rl_model_best.pt")
        torch.save({
            "iteration": it,
            "model_state_dict": current_model.state_dict(),
            "win_rate": win_rate,
            "promoted": promoted,
            "config": {"in_channels": 7, "num_channels": 64, "num_res_blocks": 5}
        }, iter_ckpt_path)

        if promoted:
            torch.save({
                "iteration": it,
                "model_state_dict": current_model.state_dict(),
                "config": {"in_channels": 7, "num_channels": 64, "num_res_blocks": 5}
            }, best_ckpt_path)

        rl_logs.append({
            "iteration": it,
            "new_samples": new_samples,
            "buffer_size": len(replay_buffer),
            "arena_win_rate": win_rate,
            "promoted": promoted,
            "time_sec": iter_elapsed
        })

    with open(os.path.join(output_dir, "rl_training_log.json"), "w", encoding="utf-8") as f:
        json.dump(rl_logs, f, indent=2)

    print("\nSelf-Play RL Complete!")
    print(f"Final RL model saved at: {os.path.join(output_dir, 'rl_model_best.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="models/checkpoints/sl_model_best.pt")
    parser.add_argument("--out", type=str, default="models/checkpoints")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--sims", type=int, default=50)
    args = parser.parse_args()

    run_self_play_rl(args.base, args.out, iterations=args.iters, games_per_iter=args.games, mcts_sims=args.sims)
