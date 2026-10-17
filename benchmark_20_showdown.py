"""
20-Game Official Showdown: Grandmaster AlphaOmok Supreme vs FriendAgent
Runs 20 full tournament games alternating Black and White (10 games as Black, 10 games as White).
Records moves, turn count, time elapsed, and final victory statistics.
"""

import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, 'C:/Users/pc/Downloads')
sys.path.insert(0, 'C:/Users/pc/omok_project')

from engine.board import Board
from friend_agent import AgentPlayer
from players.grandmaster_player import GrandmasterPlayer


def play_single_game(game_idx: int, black_is_alphaomok: bool, max_turns: int = 100):
    b = Board()
    if black_is_alphaomok:
        p_black = GrandmasterPlayer(0)
        p_white = AgentPlayer(1)
        names = ("AlphaOmok (Black)", "FriendAgent (White)")
    else:
        p_black = AgentPlayer(0)
        p_white = GrandmasterPlayer(1)
        names = ("FriendAgent (Black)", "AlphaOmok (White)")

    time_left = [60000.0, 60000.0]  # 60s clock
    t_game_start = time.perf_counter()

    for turn in range(1, max_turns + 1):
        curr = (turn - 1) % 2
        p = p_black if curr == 0 else p_white

        t0 = time.perf_counter()
        move = p.take_turn(b.grid, time_left[curr])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        time_left[curr] -= elapsed_ms

        if time_left[curr] <= 0:
            winner = 1 - curr
            w_agent = "AlphaOmok" if (winner == 0 and black_is_alphaomok) or (winner == 1 and not black_is_alphaomok) else "FriendAgent"
            dur = time.perf_counter() - t_game_start
            return {
                "game": game_idx,
                "winner": winner,
                "winner_agent": w_agent,
                "black_agent": "AlphaOmok" if black_is_alphaomok else "FriendAgent",
                "turns": turn,
                "reason": "Timeout",
                "duration": dur,
                "time_left": time_left
            }

        caps = b.place_stone(move[0], move[1], curr)
        over, winner = b.check_game_status()
        if over:
            w_agent = "AlphaOmok" if (winner == 0 and black_is_alphaomok) or (winner == 1 and not black_is_alphaomok) else "FriendAgent"
            dur = time.perf_counter() - t_game_start
            return {
                "game": game_idx,
                "winner": winner,
                "winner_agent": w_agent,
                "black_agent": "AlphaOmok" if black_is_alphaomok else "FriendAgent",
                "turns": turn,
                "reason": "5-in-a-row / Win condition",
                "duration": dur,
                "time_left": time_left
            }

    dur = time.perf_counter() - t_game_start
    return {
        "game": game_idx,
        "winner": -1,
        "winner_agent": "Draw",
        "black_agent": "AlphaOmok" if black_is_alphaomok else "FriendAgent",
        "turns": max_turns,
        "reason": "Max turns reached",
        "duration": dur,
        "time_left": time_left
    }


def main():
    total_games = 20
    print("=" * 70)
    print("    ALPHAOMOK GRANDMASTER SUPREME vs FRIENDAGENT : 20-GAME SHOWDOWN")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Device: {device}")
    print("Tournament setup ready. Starting 20 official showdown matches!\n")

    results = []
    alpha_wins = 0
    friend_wins = 0
    draws = 0

    alpha_black_wins = 0
    alpha_white_wins = 0
    friend_black_wins = 0
    friend_white_wins = 0

    for i in range(1, total_games + 1):
        # Alternate colors: Odd games AlphaOmok is Black, Even games FriendAgent is Black
        black_is_alpha = (i % 2 == 1)
        role_str = "AlphaOmok (Black) vs FriendAgent (White)" if black_is_alpha else "FriendAgent (Black) vs AlphaOmok (White)"
        print(f"--- Game {i:02d}/{total_games:02d}: {role_str} ---")

        res = play_single_game(i, black_is_alpha)
        results.append(res)

        w = res['winner_agent']
        if w == "AlphaOmok":
            alpha_wins += 1
            if black_is_alpha:
                alpha_black_wins += 1
            else:
                alpha_white_wins += 1
        elif w == "FriendAgent":
            friend_wins += 1
            if not black_is_alpha:
                friend_black_wins += 1
            else:
                friend_white_wins += 1
        else:
            draws += 1

        print(f"  -> Result: {w} WON in {res['turns']} turns ({res['duration']:.2f}s) [{res['reason']}]")
        print(f"  -> Scoreboard: AlphaOmok {alpha_wins}W - {friend_wins}L - {draws}D\n", flush=True)

    print("=" * 70)
    print("                    FINAL 20-GAME SUMMARY REPORT")
    print("=" * 70)
    print(f"Total Games Played: {total_games}")
    print(f"Grandmaster AlphaOmok: {alpha_wins} wins ({alpha_wins / total_games * 100:.1f}%)")
    print(f"FriendAgent:           {friend_wins} wins ({friend_wins / total_games * 100:.1f}%)")
    print(f"Draws:                 {draws} ({draws / total_games * 100:.1f}%)")
    print("-" * 70)
    print(f"AlphaOmok as Black:    {alpha_black_wins}/10 wins ({alpha_black_wins * 10:.1f}%)")
    print(f"AlphaOmok as White:    {alpha_white_wins}/10 wins ({alpha_white_wins * 10:.1f}%)")
    print(f"FriendAgent as Black:  {friend_black_wins}/10 wins ({friend_black_wins * 10:.1f}%)")
    print(f"FriendAgent as White:  {friend_white_wins}/10 wins ({friend_white_wins * 10:.1f}%)")
    print("=" * 70)

    # Save detailed results to logs
    os.makedirs('logs/showdown', exist_ok=True)
    out_file = 'logs/showdown/20_game_showdown_supreme_results.txt'
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("Game,Black,White,Winner,Turns,Duration(s),Reason\n")
        for r in results:
            black = "AlphaOmok" if r['black_agent'] == "AlphaOmok" else "FriendAgent"
            white = "FriendAgent" if black == "AlphaOmok" else "AlphaOmok"
            f.write(f"{r['game']},{black},{white},{r['winner_agent']},{r['turns']},{r['duration']:.2f},{r['reason']}\n")
        f.write(f"\nFinal: AlphaOmok {alpha_wins} - {friend_wins} FriendAgent (Draws: {draws})\n")
    print(f"Detailed game-by-game log saved to: {out_file}")


if __name__ == '__main__':
    main()
