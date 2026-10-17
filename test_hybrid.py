import sys
import time

sys.path.insert(0, 'C:/Users/pc/Downloads')
sys.path.insert(0, 'C:/Users/pc/omok_project')

from engine.board import Board
from friend_agent import AgentPlayer
from players.grandmaster_player import GrandmasterPlayer as GrandmasterAlphaOmok


def run_match(black_is_alphaomok=True):
    b = Board()
    if black_is_alphaomok:
        p_black = GrandmasterAlphaOmok(0)
        p_white = AgentPlayer(1)
        names = ("Grandmaster AlphaOmok (Black)", "FriendAgent (White)")
    else:
        p_black = AgentPlayer(0)
        p_white = GrandmasterAlphaOmok(1)
        names = ("FriendAgent (Black)", "Grandmaster AlphaOmok (White)")

    print(f"\n{'='*60}")
    print(f"MATCH: {names[0]} vs {names[1]}")
    print(f"{'='*60}")

    curr = 0
    for turn in range(1, 60):
        p = p_black if curr == 0 else p_white
        t_start = time.perf_counter()
        mv = p.take_turn(b.grid, 60000.0)
        dur = (time.perf_counter() - t_start) * 1000.0
        caps = b.place_stone(mv[0], mv[1], curr)
        cap_str = f" [Caps: {len(caps)}]" if caps else ""
        name = "AlphaOmok" if (curr == 0 and black_is_alphaomok) or (curr == 1 and not black_is_alphaomok) else "Friend"
        print(f"Turn {turn:2d}: {name} -> {mv}{cap_str} ({dur:.1f}ms)", flush=True)
        over, winner = b.check_game_status()
        if over:
            w_name = names[winner] if winner in (0, 1) else "Draw"
            print(f"\n>>> GAME OVER at Turn {turn}: WINNER is {w_name}! <<<")
            return winner
        curr = 1 - curr
    return -1


if __name__ == "__main__":
    print("--- Game 1: Grandmaster AlphaOmok (Black) vs FriendAgent (White) ---")
    w1 = run_match(black_is_alphaomok=True)

    print("\n--- Game 2: FriendAgent (Black) vs Grandmaster AlphaOmok (White) ---")
    w2 = run_match(black_is_alphaomok=False)
