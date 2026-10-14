"""
HumanPlayer implementation for Omok++.
Allows human player to enter coordinates via terminal in `n,m` format.
"""

from typing import List, Tuple
try:
    from alphaomok.players.player import Player
except ImportError:
    from players.player import Player


class HumanPlayer(Player):
    def __init__(self, player_id: int):
        super().__init__(player_id)

    def take_turn(self, board: List[List[int]], time: float) -> Tuple[int, int]:
        color_name = "Black (X)" if self.player_id == 0 else "White (O)"
        remaining_sec = time / 1000.0 if time >= 0 else float('inf')

        if time >= 0:
            print(f"[{color_name}] Your turn! Remaining time: {remaining_sec:.1f}s ({time:.0f} ms)")
        else:
            print(f"[{color_name}] Your turn! (No time limit)")

        while True:
            try:
                user_input = input("Enter move (row,col e.g., 9,9): ").strip()
                parts = user_input.split(',')
                if len(parts) != 2:
                    print("Invalid format. Please enter as n,m with no spaces (e.g., 9,9).")
                    continue
                r, c = int(parts[0]), int(parts[1])
                if not (0 <= r < 19 and 0 <= c < 19):
                    print(f"Coordinates ({r}, {c}) are out of bounds (0-18). Try again.")
                    continue
                if board[r][c] != -1:
                    print(f"Position ({r}, {c}) is already occupied. Try again.")
                    continue
                return (r, c)
            except ValueError:
                print("Invalid numbers. Please enter integers as n,m (e.g., 9,9).")
            except (KeyboardInterrupt, EOFError):
                print("\nInput cancelled.")
                raise
