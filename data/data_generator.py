"""
Omok++ Synthetic Dataset Generator.
Generates training samples (board_state, current_player, policy_target, game_outcome)
using HeuristicSearchPlayer and SimplePlayer matchups.
Features:
- Opening randomization (temperature exploration) for pattern diversity
- Game-level Train/Validation splitting (prevents data leakage)
- 8-fold Dihedral (D4) symmetry augmentation (rotations & reflections)
"""

import os
import sys
import random
import time
import numpy as np
from typing import List, Tuple, Dict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.board import Board
from models.policy_value_net import PolicyValueNet


def play_synthetic_game(agent_type0: str, agent_type1: str, opening_random_moves: int = 4) -> List[Tuple[np.ndarray, int, int, int]]:
    """
    Plays a single synthetic game.
    Returns list of tuples: (state_tensor_7x19x19, current_player, move_idx_0_to_360, winner)
    """
    b = Board()
    history: List[Tuple[np.ndarray, int, int]] = [] # (state_tensor, player, move_idx)

    # Instantiate players
    def make_player(atype: str, pid: int):
        from players.simpleplayer import SimplePlayer
        from players.heuristic_search_player import HeuristicSearchPlayer
        if atype == 'heuristic':
            return HeuristicSearchPlayer(pid, max_depth=1, time_budget_ms=100.0)
        elif atype == 'simple':
            return SimplePlayer(pid)
        else:
            return SimplePlayer(pid)

    p0 = make_player(agent_type0, 0)
    p1 = make_player(agent_type1, 1)
    players = {0: p0, 1: p1}

    curr = Board.BLACK
    move_count = 0

    while True:
        legal = b.get_legal_moves()
        if not legal:
            winner = -1
            break

        state_tensor = PolicyValueNet.encode_board(b, curr)

        # Opening randomization: pick from near-center candidate moves
        if move_count < opening_random_moves:
            # Pick from center radius 3
            center_moves = [(r, c) for r, c in legal if abs(r - 9) <= 3 and abs(c - 9) <= 3]
            if center_moves and random.random() < 0.65:
                move = random.choice(center_moves)
            else:
                move = players[curr].take_turn(b.get_grid(), 60000.0)
        else:
            move = players[curr].take_turn(b.get_grid(), 60000.0)

        r, c = move
        move_idx = r * 19 + c
        history.append((state_tensor, curr, move_idx))

        b.place_stone(r, c, curr)
        move_count += 1

        is_over, winner = b.check_game_status()
        if is_over:
            break

        curr = 1 - curr

    # Build annotated samples: (state, player, move_idx, outcome_for_player)
    # outcome: +1.0 (win), -1.0 (loss), 0.0 (draw)
    annotated = []
    for state_tensor, player, move_idx in history:
        if winner == -1 or winner is None:
            outcome = 0.0
        elif winner == player:
            outcome = 1.0
        else:
            outcome = -1.0
        annotated.append((state_tensor, player, move_idx, outcome))

    return annotated


def augment_8fold(state: np.ndarray, move_idx: int) -> List[Tuple[np.ndarray, int]]:
    """
    Applies 8 dihedral symmetries (4 rotations x 2 horizontal flips) to state (7, 19, 19)
    and move_idx (0..360).
    Returns list of 8 (augmented_state, augmented_move_idx).
    """
    results = []
    move_grid = np.zeros((19, 19), dtype=np.float32)
    mr, mc = divmod(move_idx, 19)
    move_grid[mr, mc] = 1.0

    for rot in range(4):
        # Rotate state (along axes 1, 2)
        rot_state = np.rot90(state, k=rot, axes=(1, 2))
        rot_move = np.rot90(move_grid, k=rot, axes=(0, 1))

        # Identity
        idx1 = int(np.argmax(rot_move))
        results.append((rot_state.copy(), idx1))

        # Flip horizontally
        flip_state = np.flip(rot_state, axis=2)
        flip_move = np.fliplr(rot_move)
        idx2 = int(np.argmax(flip_move))
        results.append((flip_state.copy(), idx2))

    return results


def generate_dataset(num_games: int, output_dir: str, val_ratio: float = 0.15):
    """
    Generates synthetic games, splits train/val at the game level,
    applies 8-fold augmentation, and saves as .npz files.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {num_games} synthetic games...")

    matchup_types = [
        ('heuristic', 'heuristic'),
        ('heuristic', 'simple'),
        ('simple', 'heuristic'),
        ('simple', 'simple')
    ]

    all_game_data: List[List[Tuple[np.ndarray, int, int, float]]] = []

    start_t = time.time()
    for g in range(num_games):
        m0, m1 = random.choice(matchup_types)
        game_samples = play_synthetic_game(m0, m1, opening_random_moves=random.randint(2, 5))
        all_game_data.append(game_samples)
        if (g + 1) % 20 == 0 or g == num_games - 1:
            print(f"[{g+1}/{num_games}] games completed ({time.time() - start_t:.1f}s)")

    # Split train and val by game (preventing data leakage across moves within the same game)
    random.shuffle(all_game_data)
    val_count = max(1, int(num_games * val_ratio))
    val_games = all_game_data[:val_count]
    train_games = all_game_data[val_count:]

    def process_and_augment(games):
        states, policies, values = [], [], []
        for g_samples in games:
            for state, player, move_idx, outcome in g_samples:
                # Augment
                augmented = augment_8fold(state, move_idx)
                for aug_state, aug_move_idx in augmented:
                    policy_target = np.zeros(361, dtype=np.float32)
                    policy_target[aug_move_idx] = 1.0

                    states.append(aug_state)
                    policies.append(policy_target)
                    values.append(outcome)
        return (np.array(states, dtype=np.float32),
                np.array(policies, dtype=np.float32),
                np.array(values, dtype=np.float32))

    print("Augmenting and formatting training set...")
    train_states, train_policies, train_values = process_and_augment(train_games)
    print(f"Train samples: {len(train_states)} (from {len(train_games)} games)")

    print("Augmenting and formatting validation set...")
    val_states, val_policies, val_values = process_and_augment(val_games)
    print(f"Val samples: {len(val_states)} (from {len(val_games)} games)")

    train_path = os.path.join(output_dir, "train_data.npz")
    val_path = os.path.join(output_dir, "val_data.npz")

    np.savez_compressed(train_path, states=train_states, policies=train_policies, values=train_values)
    np.savez_compressed(val_path, states=val_states, policies=val_policies, values=val_values)

    print(f"Saved dataset to {train_path} and {val_path}")
    print(f"Total positions generated: {len(train_states) + len(val_states)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=150, help="Number of games to generate")
    parser.add_argument("--out", type=str, default="data/dataset", help="Output directory")
    args = parser.parse_args()

    generate_dataset(args.games, args.out)
