"""
SimplePlayer implementation for Omok++.
Rule-based heuristic player that:
1. Detects immediate winning moves (5-in-a-row).
2. Blocks opponent's immediate winning moves.
3. Finds stone-capturing moves.
4. Defends against immediate capture threats.
5. Evaluates open-fours, open-threes, and positional proximity.
"""

from typing import List, Tuple, Set
import random
from players.player import Player
from engine.board import Board


class SimplePlayer(Player):
    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.opponent_id = 1 - player_id

    def take_turn(self, board: List[List[int]], time: float) -> Tuple[int, int]:
        sim_board = Board(board)
        legal_moves = sim_board.get_legal_moves()

        if not legal_moves:
            raise RuntimeError("No legal moves available.")

        # Opening: if center (9, 9) is empty on first move, play it
        if len(legal_moves) >= 360 and sim_board.is_empty(9, 9):
            return (9, 9)

        # 1. Check if we can win immediately with a 5-in-a-row
        for r, c in legal_moves:
            b_copy = sim_board.copy()
            b_copy.place_stone(r, c, self.player_id)
            is_over, winner = b_copy.check_game_status()
            if is_over and winner == self.player_id:
                return (r, c)

        # 2. Check if opponent can win immediately on their next move -> block it!
        opponent_winning_moves = []
        for r, c in legal_moves:
            b_copy = sim_board.copy()
            b_copy.place_stone(r, c, self.opponent_id)
            is_over, winner = b_copy.check_game_status()
            if is_over and winner == self.opponent_id:
                opponent_winning_moves.append((r, c))
        if opponent_winning_moves:
            return opponent_winning_moves[0]

        # 3. Look for capturing moves
        capture_moves = []
        for r, c in legal_moves:
            b_copy = sim_board.copy()
            captured = b_copy.place_stone(r, c, self.player_id)
            if len(captured) > 0:
                is_over, winner = b_copy.check_game_status()
                # Ensure capture didn't hand victory to opponent
                if not (is_over and winner == self.opponent_id):
                    capture_moves.append(((r, c), len(captured)))
        if capture_moves:
            # Sort by number of captured stones
            capture_moves.sort(key=lambda x: x[1], reverse=True)
            return capture_moves[0][0]

        # 4. Score moves near existing stones
        # Candidate generation: only cells within Chebyshev distance 2 of existing stones
        active_cells: Set[Tuple[int, int]] = set()
        has_stones = False
        for r in range(Board.BOARD_SIZE):
            for c in range(Board.BOARD_SIZE):
                if board[r][c] != Board.EMPTY:
                    has_stones = True
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = r + dr, c + dc
                            if sim_board.is_empty(nr, nc):
                                active_cells.add((nr, nc))

        candidates = list(active_cells) if (has_stones and active_cells) else legal_moves

        best_score = -float('inf')
        best_moves = []

        for r, c in candidates:
            score = self._evaluate_move(sim_board, r, c)
            if score > best_score:
                best_score = score
                best_moves = [(r, c)]
            elif score == best_score:
                best_moves.append((r, c))

        return random.choice(best_moves)

    def _evaluate_move(self, board: Board, r: int, c: int) -> float:
        """Heuristic evaluation for placing a stone at (r, c)."""
        score = 0.0

        # Center preference
        center_dist = abs(r - 9) + abs(c - 9)
        score -= center_dist * 0.1

        # Evaluate potential lines for player and opponent
        for dr, dc in Board.AXES:
            # Own pattern score
            score += self._score_line(board, r, c, dr, dc, self.player_id) * 1.0
            # Opponent pattern score (blocking opponent threats)
            score += self._score_line(board, r, c, dr, dc, self.opponent_id) * 0.85

        return score

    def _score_line(self, board: Board, r: int, c: int, dr: int, dc: int, player: int) -> float:
        """Scores consecutive friendly stones in this direction."""
        # Count consecutive friendly stones in positive direction
        count_pos = 0
        step = 1
        while board.is_valid_coord(r + step * dr, c + step * dc) and board.grid[r + step * dr][c + step * dc] == player:
            count_pos += 1
            step += 1
        open_pos = board.is_empty(r + step * dr, c + step * dc)

        # Count consecutive friendly stones in negative direction
        count_neg = 0
        step = 1
        while board.is_valid_coord(r - step * dr, c - step * dc) and board.grid[r - step * dr][c - step * dc] == player:
            count_neg += 1
            step += 1
        open_neg = board.is_empty(r - step * dr, c - step * dc)

        total_stones = count_pos + count_neg + 1
        open_ends = (1 if open_pos else 0) + (1 if open_neg else 0)

        # 5-in-a-row is huge
        if total_stones == 5:
            return 10000.0
        # 6-in-a-row is NOT a win in Omok++ (can even be dangerous)
        elif total_stones >= 6:
            return -50.0
        elif total_stones == 4:
            if open_ends == 2:
                return 3000.0  # Open 4
            elif open_ends == 1:
                return 800.0   # Blocked 4
        elif total_stones == 3:
            if open_ends == 2:
                return 500.0   # Open 3
            elif open_ends == 1:
                return 80.0
        elif total_stones == 2:
            if open_ends == 2:
                return 40.0
            elif open_ends == 1:
                return 10.0

        return 1.0
