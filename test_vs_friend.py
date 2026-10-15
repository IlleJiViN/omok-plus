import sys
import time

sys.path.insert(0, 'C:/Users/pc/omok_project')
sys.path.insert(0, 'C:/Users/pc/Downloads')

from engine.board import Board
from players.smart_player import SmartPlayer
from friend_agent import AgentPlayer

def run_match(black_is_alphaomok=True, time_limit_sec=60.0):
    board = Board()
    if black_is_alphaomok:
        p_black = SmartPlayer(0)
        p_white = AgentPlayer(1)
        names = ("AlphaOmok (Black)", "FriendAgent (White)")
    else:
        p_black = AgentPlayer(0)
        p_white = SmartPlayer(1)
        names = ("FriendAgent (Black)", "AlphaOmok (White)")

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
    print("--- Game 1: AlphaOmok (Black) vs FriendAgent (White) ---")
    w1 = run_match(black_is_alphaomok=True)
    
    print("\n--- Game 2: FriendAgent (Black) vs AlphaOmok (White) ---")
    w2 = run_match(black_is_alphaomok=False)
