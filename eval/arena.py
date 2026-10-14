"""
Omok++ Evaluation Arena.
Runs head-to-head tournaments between any pairs of agents.
Tracks:
- Black win rate, White win rate, Draw rate
- Move times (average, max, std)
- Total match duration and timeouts
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.board import Board
from engine.gamemanager import GameManager
from players.player import Player
from players.randomplayer import RandomPlayer
from players.simpleplayer import SimplePlayer
from players.heuristic_search_player import HeuristicSearchPlayer
from players.sl_cnn_player import SLCNNPlayer
from players.mcts_player import MCTSPlayer
from players.rl_mcts_player import RLMCTSPlayer
from players.smart_player import SmartPlayer


def run_head_to_head(player_a_fn,
                     player_b_fn,
                     name_a: str,
                     name_b: str,
                     num_games: int = 8,
                     time_limit_minutes: int = 1) -> Dict[str, Any]:
    """
    Plays num_games between Agent A and Agent B.
    Alternates colors: exactly num_games//2 with A as Black, num_games//2 with B as Black.
    """
    a_wins = 0
    b_wins = 0
    draws = 0
    a_black_wins = 0
    b_black_wins = 0
    a_white_wins = 0
    b_white_wins = 0

    a_times = []
    b_times = []

    print(f"\n--- Matchup: {name_a} vs {name_b} ({num_games} games, {time_limit_minutes}m time limit) ---")

    for g in range(num_games):
        a_is_black = (g % 2 == 0)

        if a_is_black:
            p0 = player_a_fn(Board.BLACK)
            p1 = player_b_fn(Board.WHITE)
        else:
            p0 = player_b_fn(Board.BLACK)
            p1 = player_a_fn(Board.WHITE)

        gm = GameManager(p0, p1, time_limit_minutes=time_limit_minutes, verbose=False)
        t_start = time.perf_counter()
        winner, reason = gm.play_game()
        match_time = time.perf_counter() - t_start

        # Record outcome
        if winner == -1:
            draws += 1
            res_str = "Draw"
        elif (winner == Board.BLACK and a_is_black) or (winner == Board.WHITE and not a_is_black):
            a_wins += 1
            res_str = f"{name_a} Win"
            if a_is_black:
                a_black_wins += 1
            else:
                a_white_wins += 1
        else:
            b_wins += 1
            res_str = f"{name_b} Win"
            if not a_is_black:
                b_black_wins += 1
            else:
                b_white_wins += 1

        color_str = f"{name_a} (Black) vs {name_b} (White)" if a_is_black else f"{name_b} (Black) vs {name_a} (White)"
        print(f"  Game {g+1}/{num_games}: [{color_str}] -> {res_str} ({match_time:.2f}s)")

    decisive = a_wins + b_wins
    a_win_rate = (a_wins / max(1, num_games)) * 100.0
    b_win_rate = (b_wins / max(1, num_games)) * 100.0
    draw_rate = (draws / max(1, num_games)) * 100.0

    summary = {
        "agent_a": name_a,
        "agent_b": name_b,
        "num_games": num_games,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_win_rate": a_win_rate,
        "b_win_rate": b_win_rate,
        "draw_rate": draw_rate,
        "a_black_wins": a_black_wins,
        "a_white_wins": a_white_wins,
        "b_black_wins": b_black_wins,
        "b_white_wins": b_white_wins,
    }

    print(f"Result: {name_a} {a_wins}W - {b_wins}L - {draws}D (Win Rate: {a_win_rate:.1f}%)")
    return summary
