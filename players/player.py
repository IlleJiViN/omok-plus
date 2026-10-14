"""
Abstract Base Class for Omok++ Players.
Follows CAS 1102 Project 1 specification:
- take_turn(board, time) -> (row, col)
  board: 19x19 2D list of int (-1: empty, 0: black, 1: white)
  time: remaining time in milliseconds (int/float)
"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class Player(ABC):
    def __init__(self, player_id: int):
        """
        player_id: 0 for Black (Player 1), 1 for White (Player 2).
        """
        self.player_id = player_id

    @abstractmethod
    def take_turn(self, board: List[List[int]], time: float) -> Tuple[int, int]:
        """
        Receives current board state and remaining time (ms).
        Returns (row, col) move.
        Do NOT modify `board`.
        """
        pass
