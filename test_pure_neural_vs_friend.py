"""
Test PureNeuralPlayer (100% Neural Network) vs FriendAgent (Rule-based Expert).
Verifies if the fine-tuned neural network alone, without any handcrafted heuristics,
can detect tactical threats, block forks, and defeat FriendAgent.
"""

import sys
import time

sys.path.insert(0, 'C:/Users/pc/omok_project')
sys.path.insert(0, 'C:/Users/pc/Downloads')

from engine.board import Board
from players.pure_neural_player import PureNeuralPlayer
from friend_agent import AgentPlayer


def run_match(black_is_neural=True, time_limit_sec=60.0, ckpt_path=None):
    board = Board()
    if black_is_neural:
        p_black = PureNeuralPlayer(0, ckpt_path=ckpt_path, n_sims=50)
        p_white = AgentPlayer(1)
        names = ("PureNeural (Black)", "FriendAgent (White)")
    else:
        p_black = AgentPlayer(0)
        p_white = PureNeuralPlayer(1, ckpt_path=ckpt_path, n_sims=50)
        names = ("FriendAgent (Black)", "PureNeural (White)")

    time_left = [time_limit_sec * 1000.0, time_limit_sec * 1000.0]
    turn_cnt = 0
    print(f"\n{'='*60}")
    print(f"MATCH: {names[0]} vs {names[1]} (Time Limit: {time_limit_sec}s)")
    print(f"{'='*60}")

    while True:
        curr = turn_cnt % 2
        player = p_black if curr == 0 else p_white
        t_start = time.perf_counter()

        move = player.take_turn(board.get_grid(), time_left[curr])
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        time_left[curr] -= elapsed_ms
        turn_cnt += 1

        if time_left[curr] <= 0:
            print(f"Turn {turn_cnt}: Player {curr} TIMEOUT! Winner is {1 - curr}")
            return 1 - curr

        r, c = move
        captured = board.place_stone(r, c, curr)
        cap_str = f" [Captured {len(captured)} stones]" if captured else ""
        print(f"Turn {turn_cnt:2d}: {'Black' if curr == 0 else 'White'} -> ({r:2d}, {c:2d}){cap_str} (took {elapsed_ms:.1f}ms, {time_left[curr]/1000:.1f}s left)")

        is_over, winner = board.check_game_status()
        if is_over:
            w_name = names[winner] if winner in (0, 1) else "Draw"
            print(f"\nGAME OVER at Turn {turn_cnt}: WINNER is {w_name}!")
            return winner


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="models/checkpoints/rl_model_pure_neural.pt")
    args = parser.parse_args()

    print(f"--- Benchmark with Checkpoint: {args.ckpt} ---")
    print("\n--- Game 1: PureNeural (Black) vs FriendAgent (White) ---")
    w1 = run_match(black_is_neural=True, ckpt_path=args.ckpt)

    print("\n--- Game 2: FriendAgent (Black) vs PureNeural (White) ---")
    w2 = run_match(black_is_neural=False, ckpt_path=args.ckpt)
