"""
Heuristic Search Player (Agent A) for Omok++.
Implements an Alpha-Beta Minimax search with:
- Immediate win and loss detection (tactical overrides)
- Capture detection and capture threat evaluation
- Move ordering based on local pattern scoring
- Candidate filtering (radius 2 of existing stones)
- Iterative deepening with time control
"""

import time
from typing import List, Tuple, Set, Optional
try:
    from alphaomok.players.player import Player
    from alphaomok.engine.board import Board
except ImportError:
    from players.player import Player
    from engine.board import Board


class HeuristicSearchPlayer(Player):
    def __init__(self, player_id: int, max_depth: int = 2, time_budget_ms: float = 800.0):
        super().__init__(player_id)
        self.opponent_id = 1 - player_id
        self.max_depth = max_depth
        self.time_budget_ms = time_budget_ms

    def take_turn(self, board: List[List[int]], time_left_ms: float) -> Tuple[int, int]:
        sim_board = Board(board)
        legal_moves = sim_board.get_legal_moves()

        if not legal_moves:
            raise RuntimeError("No legal moves available.")

        # Opening: first move play center
        if len(legal_moves) >= 360 and sim_board.is_empty(9, 9):
            return (9, 9)

        start_time = time.perf_counter()

        # Allocate turn time budget dynamically based on remaining time
        if time_left_ms > 0:
            # If remaining time is tight, scale down budget
            budget = min(self.time_budget_ms, max(50.0, time_left_ms / 30.0))
        else:
            budget = self.time_budget_ms

        deadline = start_time + (budget / 1000.0)

        # 1. Immediate tactical checks (depth 1)
        # Immediate winning move
        for r, c in legal_moves:
            b_copy = sim_board.copy()
            b_copy.place_stone(r, c, self.player_id)
            is_over, winner = b_copy.check_game_status()
            if is_over and winner == self.player_id:
                return (r, c)

        # Immediate opponent win to block
        opp_wins = []
        for r, c in legal_moves:
            b_copy = sim_board.copy()
            b_copy.place_stone(r, c, self.opponent_id)
            is_over, winner = b_copy.check_game_status()
            if is_over and winner == self.opponent_id:
                opp_wins.append((r, c))
        if opp_wins:
            return opp_wins[0]

        # 2. Candidate generation: cells within radius 2 of existing stones
        candidates = self._get_candidates(sim_board, legal_moves)

        # Move ordering heuristic
        scored_candidates = []
        for r, c in candidates:
            score = self._quick_move_score(sim_board, r, c, self.player_id)
            scored_candidates.append((score, (r, c)))
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        ordered_candidates = [move for _, move in scored_candidates[:25]] # top 25

        best_move = ordered_candidates[0]

        # Iterative deepening Alpha-Beta
        for depth in range(1, self.max_depth + 1):
            if time.perf_counter() >= deadline:
                break

            current_best_move = None
            alpha = -float('inf')
            beta = float('inf')

            for move in ordered_candidates:
                if time.perf_counter() >= deadline:
                    break

                b_copy = sim_board.copy()
                b_copy.place_stone(move[0], move[1], self.player_id)
                is_over, winner = b_copy.check_game_status()

                if is_over:
                    if winner == self.player_id:
                        return move
                    elif winner == -1:
                        val = 0.0
                    else:
                        val = -100000.0
                else:
                    val = self._min_val(b_copy, depth - 1, alpha, beta, deadline)

                if val > alpha:
                    alpha = val
                    current_best_move = move

            if current_best_move is not None:
                best_move = current_best_move

        return best_move

    def _min_val(self, board: Board, depth: int, alpha: float, beta: float, deadline: float) -> float:
        if depth <= 0 or time.perf_counter() >= deadline:
            return self._evaluate_board(board)

        candidates = self._get_candidates(board, board.get_legal_moves())
        scored_moves = []
        for r, c in candidates:
            score = self._quick_move_score(board, r, c, self.opponent_id)
            scored_moves.append((score, (r, c)))
        scored_moves.sort(key=lambda x: x[0], reverse=True)

        v = float('inf')
        for _, move in scored_moves[:15]:
            b_copy = board.copy()
            b_copy.place_stone(move[0], move[1], self.opponent_id)
            is_over, winner = b_copy.check_game_status()

            if is_over:
                if winner == self.opponent_id:
                    return -100000.0
                elif winner == -1:
                    val = 0.0
                else:
                    val = 100000.0
            else:
                val = self._max_val(b_copy, depth - 1, alpha, beta, deadline)

            v = min(v, val)
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def _max_val(self, board: Board, depth: int, alpha: float, beta: float, deadline: float) -> float:
        if depth <= 0 or time.perf_counter() >= deadline:
            return self._evaluate_board(board)

        candidates = self._get_candidates(board, board.get_legal_moves())
        scored_moves = []
        for r, c in candidates:
            score = self._quick_move_score(board, r, c, self.player_id)
            scored_moves.append((score, (r, c)))
        scored_moves.sort(key=lambda x: x[0], reverse=True)

        v = -float('inf')
        for _, move in scored_moves[:15]:
            b_copy = board.copy()
            b_copy.place_stone(move[0], move[1], self.player_id)
            is_over, winner = b_copy.check_game_status()

            if is_over:
                if winner == self.player_id:
                    return 100000.0
                elif winner == -1:
                    val = 0.0
                else:
                    val = -100000.0
            else:
                val = self._min_val(b_copy, depth - 1, alpha, beta, deadline)

            v = max(v, val)
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def _get_candidates(self, board: Board, legal_moves: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        active: Set[Tuple[int, int]] = set()
        has_stones = False
        for r in range(Board.BOARD_SIZE):
            for c in range(Board.BOARD_SIZE):
                if board.grid[r][c] != Board.EMPTY:
                    has_stones = True
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            nr, nc = r + dr, c + dc
                            if board.is_empty(nr, nc):
                                active.add((nr, nc))
        return list(active) if (has_stones and active) else legal_moves

    def _quick_move_score(self, board: Board, r: int, c: int, player: int) -> float:
        opp = 1 - player
        score = 0.0
        # Check captures
        for dr, dc in Board.DIRECTIONS:
            r1, c1 = r + dr, c + dc
            r2, c2 = r + 2 * dr, c + 2 * dc
            r3, c3 = r + 3 * dr, c + 3 * dc
            if (board.is_valid_coord(r1, c1) and board.is_valid_coord(r2, c2) and board.is_valid_coord(r3, c3)):
                if board.grid[r1][c1] == opp and board.grid[r2][c2] == opp and board.grid[r3][c3] == player:
                    score += 600.0

        for dr, dc in Board.AXES:
            score += self._line_len_score(board, r, c, dr, dc, player) * 1.0
            score += self._line_len_score(board, r, c, dr, dc, opp) * 0.85
        return score

    def _line_len_score(self, board: Board, r: int, c: int, dr: int, dc: int, player: int) -> float:
        pos_cnt = 0
        step = 1
        while board.is_valid_coord(r + step * dr, c + step * dc) and board.grid[r + step * dr][c + step * dc] == player:
            pos_cnt += 1
            step += 1
        open_pos = board.is_empty(r + step * dr, c + step * dc)

        neg_cnt = 0
        step = 1
        while board.is_valid_coord(r - step * dr, c - step * dc) and board.grid[r - step * dr][c - step * dc] == player:
            neg_cnt += 1
            step += 1
        open_neg = board.is_empty(r - step * dr, c - step * dc)

        tot = pos_cnt + neg_cnt + 1
        open_ends = (1 if open_pos else 0) + (1 if open_neg else 0)

        if tot == 5:
            return 50000.0
        elif tot >= 6:
            return -100.0
        elif tot == 4:
            return 4000.0 if open_ends == 2 else (1200.0 if open_ends == 1 else 0.0)
        elif tot == 3:
            return 800.0 if open_ends == 2 else (150.0 if open_ends == 1 else 0.0)
        elif tot == 2:
            return 60.0 if open_ends == 2 else 15.0
        return 2.0

    def _evaluate_board(self, board: Board) -> float:
        # Perspective of self.player_id
        is_over, winner = board.check_game_status()
        if is_over:
            if winner == self.player_id:
                return 100000.0
            elif winner == self.opponent_id:
                return -100000.0
            else:
                return 0.0

        # Capture differential
        cap_diff = board.captures[self.player_id] - board.captures[self.opponent_id]
        score = cap_diff * 300.0

        # Scan board lines
        my_eval = self._evaluate_player_lines(board, self.player_id)
        opp_eval = self._evaluate_player_lines(board, self.opponent_id)

        return score + my_eval - opp_eval * 1.1

    def _evaluate_player_lines(self, board: Board, player: int) -> float:
        tot_score = 0.0
        for dr, dc in Board.AXES:
            for r in range(Board.BOARD_SIZE):
                for c in range(Board.BOARD_SIZE):
                    if board.grid[r][c] == player:
                        pr, pc = r - dr, c - dc
                        if not board.is_valid_coord(pr, pc) or board.grid[pr][pc] != player:
                            cnt = 0
                            cr, cc = r, c
                            while board.is_valid_coord(cr, cc) and board.grid[cr][cc] == player:
                                cnt += 1
                                cr += dr
                                cc += dc
                            # check ends
                            open_start = board.is_empty(pr, pc)
                            open_end = board.is_empty(cr, cc)
                            open_ends = (1 if open_start else 0) + (1 if open_end else 0)

                            if cnt == 5:
                                tot_score += 100000.0
                            elif cnt == 4:
                                tot_score += 5000.0 if open_ends == 2 else (1200.0 if open_ends == 1 else 0)
                            elif cnt == 3:
                                tot_score += 600.0 if open_ends == 2 else (100.0 if open_ends == 1 else 0)
                            elif cnt == 2:
                                tot_score += 40.0 if open_ends == 2 else 10.0
        return tot_score
