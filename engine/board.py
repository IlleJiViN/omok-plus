"""
Omok++ Board and State Representation.
Specification: 19x19 board.
Values:
  -1 : Empty
   0 : Black (Player 1, moves first)
   1 : White (Player 2)
"""

from typing import List, Tuple, Optional
import copy


class Board:
    BOARD_SIZE = 19
    EMPTY = -1
    BLACK = 0
    WHITE = 1

    # 8 directional offsets (dr, dc)
    DIRECTIONS = [
        (-1, 0),  # Up
        (1, 0),   # Down
        (0, -1),  # Left
        (0, 1),   # Right
        (-1, -1), # Up-Left
        (-1, 1),  # Up-Right
        (1, -1),  # Down-Left
        (1, 1),   # Down-Right
    ]

    # 4 line axes: Horizontal, Vertical, Diagonal (\), Anti-Diagonal (/)
    AXES = [
        (0, 1),   # Horizontal
        (1, 0),   # Vertical
        (1, 1),   # Diagonal (\)
        (1, -1),  # Anti-diagonal (/)
    ]

    def __init__(self, grid: Optional[List[List[int]]] = None):
        if grid is not None:
            self.grid = [list(row) for row in grid]
        else:
            self.grid = [[self.EMPTY for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        self.captures = {self.BLACK: 0, self.WHITE: 0} # Number of captured stones by each player
        self.last_move: Optional[Tuple[int, int]] = None
        self.move_history: List[Tuple[int, int, int]] = [] # (player, r, c)

    def copy(self) -> 'Board':
        new_b = Board(self.grid)
        new_b.captures = dict(self.captures)
        new_b.last_move = self.last_move
        new_b.move_history = list(self.move_history)
        return new_b

    def get_grid(self) -> List[List[int]]:
        """Returns 19x19 nested list with -1, 0, 1."""
        return [list(row) for row in self.grid]

    def is_valid_coord(self, r: int, c: int) -> bool:
        return 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE

    def is_empty(self, r: int, c: int) -> bool:
        return self.is_valid_coord(r, c) and self.grid[r][c] == self.EMPTY

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Returns all legal (empty) positions."""
        moves = []
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.grid[r][c] == self.EMPTY:
                    moves.append((r, c))
        return moves

    def is_full(self) -> bool:
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.grid[r][c] == self.EMPTY:
                    return False
        return True

    def place_stone(self, r: int, c: int, player: int) -> List[Tuple[int, int]]:
        """
        Places a stone for player at (r, c), executes captures according to Omok++ rules.
        Returns list of captured stone coordinates [(r1, c1), (r2, c2), ...].
        Raises ValueError if move is illegal.
        """
        if not self.is_valid_coord(r, c):
            raise ValueError(f"Move ({r}, {c}) is out of bounds.")
        if self.grid[r][c] != self.EMPTY:
            raise ValueError(f"Position ({r}, {c}) is already occupied.")

        self.grid[r][c] = player
        self.last_move = (r, c)
        self.move_history.append((player, r, c))

        # Check captures in all 8 directions
        opponent = 1 - player
        captured_stones: List[Tuple[int, int]] = []

        for dr, dc in self.DIRECTIONS:
            r1, c1 = r + dr, c + dc
            r2, c2 = r + 2 * dr, c + 2 * dc
            r3, c3 = r + 3 * dr, c + 3 * dc

            if (self.is_valid_coord(r1, c1) and self.is_valid_coord(r2, c2) and self.is_valid_coord(r3, c3)):
                if (self.grid[r1][c1] == opponent and 
                    self.grid[r2][c2] == opponent and 
                    self.grid[r3][c3] == player):
                    # Flanked two opponent stones!
                    captured_stones.append((r1, c1))
                    captured_stones.append((r2, c2))

        # Remove captured stones from board
        for cr, cc in captured_stones:
            self.grid[cr][cc] = self.EMPTY

        self.captures[player] += len(captured_stones)
        return captured_stones

    def count_exact_fives(self, player: int) -> int:
        """
        Counts the number of exact 5-in-a-rows for the given player.
        An exact 5-in-a-row is a contiguous sequence of exactly 5 stones of player
        bounded by non-player stones or board edges.
        6 or more in a row does NOT count.
        """
        fives_count = 0

        for dr, dc in self.AXES:
            # We iterate through all possible starting points for maximal contiguous segments.
            # To avoid duplicate counting along each line:
            # A line starting point is where (r - dr, c - dc) is off-board.
            # Then we scan along (dr, dc) to find maximal contiguous runs of player's stones.
            for r in range(self.BOARD_SIZE):
                for c in range(self.BOARD_SIZE):
                    # Check if (r, c) is the start of a contiguous segment of player stones in direction (dr, dc)
                    if self.grid[r][c] == player:
                        prev_r, prev_c = r - dr, c - dc
                        if not self.is_valid_coord(prev_r, prev_c) or self.grid[prev_r][prev_c] != player:
                            # (r, c) is the start of a maximal run of player's stones!
                            length = 0
                            curr_r, curr_c = r, c
                            while self.is_valid_coord(curr_r, curr_c) and self.grid[curr_r][curr_c] == player:
                                length += 1
                                curr_r += dr
                                curr_c += dc
                            if length == 5:
                                fives_count += 1
        return fives_count

    def check_game_status(self) -> Tuple[bool, Optional[int]]:
        """
        Checks the game status according to Omok++ rules.
        Returns:
          (is_over: bool, winner: Optional[int])
          winner can be:
            0 : Black wins
            1 : White wins
            -1 : Tie / Draw
            None : Game not over
        """
        black_fives = self.count_exact_fives(self.BLACK)
        white_fives = self.count_exact_fives(self.WHITE)

        if black_fives > 0 and white_fives > 0:
            return True, -1  # Simultaneous exact 5-in-a-row -> Tie
        elif black_fives > 0:
            return True, self.BLACK
        elif white_fives > 0:
            return True, self.WHITE

        if self.is_full():
            return True, -1  # Full board draw

        return False, None

    def display(self) -> str:
        """String representation of the board for CLI output."""
        header = "   " + " ".join(f"{c:2d}" for c in range(self.BOARD_SIZE))
        lines = [header]
        symbols = {self.EMPTY: '.', self.BLACK: 'X', self.WHITE: 'O'}
        for r in range(self.BOARD_SIZE):
            row_str = f"{r:2d} " + "  ".join(symbols[self.grid[r][c]] for c in range(self.BOARD_SIZE))
            lines.append(row_str)
        return "\n".join(lines)
