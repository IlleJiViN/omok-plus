"""
RandomPlayer implementation for Omok++.
Picks uniformly at random among all empty positions.
"""

import random
from typing import List, Tuple
from players.player import Player


class RandomPlayer(Player):
    def __init__(self, player_id: int):
        super().__init__(player_id)

    def take_turn(self, board: List[List[int]], time: float) -> Tuple[int, int]:
        empty_cells = []
        for r in range(len(board)):
            for c in range(len(board[r])):
                if board[r][c] == -1:
                    empty_cells.append((r, c))
        if not empty_cells:
            raise RuntimeError("No legal moves left on the board.")
        return random.choice(empty_cells)
