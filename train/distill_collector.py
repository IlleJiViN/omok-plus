"""
Distill Data Collector: Teacher-Guided Expert Dataset Generation.
Collects high-quality tactical training states from friend_agent.py (LightOfBucks).
Includes:
1. FriendAgent vs FriendAgent self-play games (expert tactical demonstration).
2. AlphaOmok vs FriendAgent sparring games with DAgger teacher labeling:
   On every turn, queries FriendAgent for the optimal ground-truth move.
3. 8-fold dihedral symmetry augmentation (D4).
4. Saves to data/distilled/teacher_data.npz.
"""

import os
import sys
import time
import random
import numpy as np
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'C:/Users/pc/Downloads')

from engine.board import Board
from models.policy_value_net import PolicyValueNet
from friend_agent import AgentPlayer
from data.data_generator import augment_8fold


def collect_expert_self_play(num_games: int = 15) -> List[Tuple[np.ndarray, int, float]]:
    """
    Plays FriendAgent vs FriendAgent self-play games.
    Returns: list of (augmented_state, move_idx, outcome)
    """
    samples = []
    print(f"\n[1/2] Generating {num_games} FriendAgent Expert Self-Play games...")
    start_t = time.time()

    for g in range(num_games):
        b = Board()
        p0 = AgentPlayer(0)
        p1 = AgentPlayer(1)
        history = [] # (state_7x19x19, player, move_idx)

        # Small opening randomization (center area) for pattern diversity
        open_rand = random.randint(1, 3)

        for turn in range(60):
            curr = turn % 2
            player = p0 if curr == 0 else p1
            legal = b.get_legal_moves()
            if not legal:
                break

            state = PolicyValueNet.encode_board(b, curr)

            if turn < open_rand and random.random() < 0.5:
                near_center = [(r, c) for r, c in legal if abs(r - 9) <= 2 and abs(c - 9) <= 2]
                move = random.choice(near_center) if near_center else player.take_turn(b.get_grid(), 60000.0)
            else:
                move = player.take_turn(b.get_grid(), 60000.0)

            r, c = move
            history.append((state, curr, r * 19 + c))
            b.place_stone(r, c, curr)

            is_over, winner = b.check_game_status()
            if is_over:
                break

        # Annotate outcomes
        for st, plyr, mv_idx in history:
            if winner is None or winner == -1:
                val = 0.0
            elif winner == plyr:
                val = 1.0
            else:
                val = -1.0

            # 8-fold augmentation
            for aug_st, aug_mv in augment_8fold(st, mv_idx):
                samples.append((aug_st, aug_mv, val))

        elapsed = time.time() - start_t
        print(f"  Game {g+1:2d}/{num_games:2d} finished in {turn+1:2d} turns (Winner: {winner}). Total samples so far: {len(samples)} ({elapsed:.1f}s)")

    return samples


def collect_dagger_sparring(num_games: int = 20) -> List[Tuple[np.ndarray, int, float]]:
    """
    AlphaOmok vs FriendAgent sparring with DAgger labeling:
    On every turn, queries the expert (FriendAgent) for the ground truth target move.
    """
    from players.smart_player import SmartPlayer
    samples = []
    print(f"\n[2/2] Generating {num_games} AlphaOmok vs FriendAgent DAgger Sparring games...")
    start_t = time.time()

    for g in range(num_games):
        b = Board()
        alpha_is_black = (g % 2 == 0)

        alpha_player = SmartPlayer(0 if alpha_is_black else 1)
        friend_player = AgentPlayer(1 if alpha_is_black else 0)

        # Expert consultant for ground truth labeling
        expert_p0 = AgentPlayer(0)
        expert_p1 = AgentPlayer(1)

        history = [] # (state_7x19x19, player, expert_move_idx)

        for turn in range(50):
            curr = turn % 2
            is_alpha = (curr == 0 and alpha_is_black) or (curr == 1 and not alpha_is_black)

            legal = b.get_legal_moves()
            if not legal:
                break

            state = PolicyValueNet.encode_board(b, curr)

            # Query expert for target move regardless of who is playing
            expert = expert_p0 if curr == 0 else expert_p1
            expert_move = expert.take_turn(b.get_grid(), 60000.0)
            expert_mv_idx = expert_move[0] * 19 + expert_move[1]
            history.append((state, curr, expert_mv_idx))

            # Actual move played on board
            if is_alpha:
                actual_move = alpha_player.take_turn(b.get_grid(), 60000.0)
            else:
                actual_move = expert_move

            b.place_stone(actual_move[0], actual_move[1], curr)

            is_over, winner = b.check_game_status()
            if is_over:
                break

        # Annotate outcomes
        for st, plyr, mv_idx in history:
            if winner is None or winner == -1:
                val = 0.0
            elif winner == plyr:
                val = 1.0
            else:
                val = -1.0

            for aug_st, aug_mv in augment_8fold(st, mv_idx):
                samples.append((aug_st, aug_mv, val))

        elapsed = time.time() - start_t
        print(f"  Sparring Game {g+1:2d}/{num_games:2d} finished in {turn+1:2d} turns (Winner: {winner}). Total samples so far: {len(samples)} ({elapsed:.1f}s)")

    return samples


def main():
    os.makedirs("data/distilled", exist_ok=True)
    out_file = "data/distilled/teacher_data.npz"

    t0 = time.time()
    samples1 = collect_expert_self_play(num_games=12)
    samples2 = collect_dagger_sparring(num_games=12)

    all_samples = samples1 + samples2
    random.shuffle(all_samples)

    states = np.array([s[0] for s in all_samples], dtype=np.float32)
    moves = np.array([s[1] for s in all_samples], dtype=np.int64)
    values = np.array([s[2] for s in all_samples], dtype=np.float32)

    # 10% validation split
    num_val = int(len(all_samples) * 0.1)
    val_states, train_states = states[:num_val], states[num_val:]
    val_moves, train_moves = moves[:num_val], moves[num_val:]
    val_values, train_values = values[:num_val], values[num_val:]

    np.savez_compressed(
        out_file,
        train_states=train_states,
        train_moves=train_moves,
        train_values=train_values,
        val_states=val_states,
        val_moves=val_moves,
        val_values=val_values
    )

    print(f"\n[DONE] Successfully saved distilled dataset to {out_file}")
    print(f"Total Augmented Samples: {len(all_samples):,}")
    print(f"Train Samples: {len(train_states):,}, Val Samples: {len(val_states):,}")
    print(f"Total time elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
