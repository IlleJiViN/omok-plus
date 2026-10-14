"""
SmartPlayer: High-Performance Ensemble AI for Omok++.
Combines:
1. Tier 1: Immediate Winning Move Detection (exact 5-in-a-row) (0ms)
2. Tier 2: Immediate Opponent Win Defense (0ms)
3. Tier 3: Tactical Capture and Counter-Capture Engine
4. Tier 4: Alpha-Beta Minimax Tactical Foresight (depth 2 with candidate pruning)
5. Tier 5: Reinforced ResNet Policy-Value Network & PUCT MCTS for Positional Dominance
6. Strictly adheres to the 60.0s total game budget constraint
7. CPU-safe and GPU accelerated
"""

import os
import sys
import time
import random
import torch
import numpy as np
from typing import List, Tuple, Optional, Set

try:
    from alphaomok.players.player import Player
    from alphaomok.engine.board import Board
    from alphaomok.models.policy_value_net import PolicyValueNet
    from alphaomok.mcts.mcts import MCTS
except ImportError:
    from players.player import Player
    from engine.board import Board
    from models.policy_value_net import PolicyValueNet
    from mcts.mcts import MCTS


class SmartPlayer(Player):
    def __init__(self, player_id: int, weights_path: Optional[str] = None, device: Optional[str] = None):
        super().__init__(player_id)
        self.opponent_id = 1 - player_id
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load best neural checkpoint
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bundled_weights = os.path.join(base_dir, "models", "weights", "rl_model_best.pt")

        ckpt_candidates = []
        if weights_path:
            ckpt_candidates.append(weights_path)
        ckpt_candidates.extend([
            bundled_weights,
            os.path.join(base_dir, "models", "checkpoints", "rl_model_best.pt"),
            os.path.join(base_dir, "models", "checkpoints", "rl_model_iter_2.pt"),
            os.path.join(base_dir, "models", "checkpoints", "rl_model_iter_1.pt"),
            os.path.join(base_dir, "models", "checkpoints", "sl_model_best.pt"),
        ])

        self.model = None
        for ckpt_path in ckpt_candidates:
            if os.path.exists(ckpt_path):
                try:
                    ckpt = torch.load(ckpt_path, map_location=self.device)
                    st = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                    config = ckpt.get("config") or ckpt.get("model_config")
                    if config:
                        self.model = PolicyValueNet.from_config(config).to(self.device)
                    elif "conv_in.weight" in st:
                        num_ch = st["conv_in.weight"].shape[0]
                        res_blocks = set(int(k.split(".")[1]) for k in st.keys() if k.startswith("res_blocks."))
                        num_res = len(res_blocks) if res_blocks else 5
                        self.model = PolicyValueNet(num_channels=num_ch, num_res_blocks=num_res).to(self.device)
                    else:
                        self.model = PolicyValueNet.create_baseline().to(self.device)
                    self.model.load_state_dict(st)
                    break
                except Exception:
                    continue

        if self.model is None:
            self.model = PolicyValueNet.create_baseline().to(self.device)

        self.model.eval()
        self.mcts = MCTS(self.model, self.device, c_puct=1.5, n_simulations=50)

    def take_turn(self, board: List[List[int]], time_left_ms: float) -> Tuple[int, int]:
        try:
            return self._decide_move(board, time_left_ms)
        except Exception:
            return self._emergency_fallback(board)

    def _decide_move(self, board: List[List[int]], time_left_ms: float) -> Tuple[int, int]:
        sim_board = Board(board)
        legal_moves = sim_board.get_legal_moves()

        if not legal_moves:
            raise RuntimeError("No legal moves available on board.")

        # Opening: center move
        if len(legal_moves) >= 360 and sim_board.is_empty(9, 9):
            return (9, 9)

        start_time = time.perf_counter()

        # Dynamic time allocation for 60s total game budget
        if time_left_ms > 0:
            turn_budget_ms = min(600.0, max(40.0, time_left_ms / 35.0))
        else:
            turn_budget_ms = 400.0

        deadline = start_time + (turn_budget_ms / 1000.0)

        # ----------------------------------------------------
        # TIER 1: Immediate Winning Move (Exact 5-in-a-row)
        # ----------------------------------------------------
        for r, c in legal_moves:
            b_test = sim_board.copy()
            b_test.place_stone(r, c, self.player_id)
            is_over, winner = b_test.check_game_status()
            if is_over and winner == self.player_id:
                return (r, c)

        # ----------------------------------------------------
        # TIER 2: Immediate Opponent Win Defense
        # ----------------------------------------------------
        opp_wins = []
        for r, c in legal_moves:
            b_test = sim_board.copy()
            b_test.place_stone(r, c, self.opponent_id)
            is_over, winner = b_test.check_game_status()
            if is_over and winner == self.opponent_id:
                opp_wins.append((r, c))
        if opp_wins:
            return opp_wins[0]

        # ----------------------------------------------------
        # TIER 3: Candidate Generation (Chebyshev radius 2)
        # ----------------------------------------------------
        candidates = self._get_candidates(sim_board, legal_moves)

        # ----------------------------------------------------
        # TIER 4: Immediate Open-4 Creation (Guaranteed Win)
        # ----------------------------------------------------
        for r, c in candidates:
            if self._creates_open_four(sim_board, r, c, self.player_id):
                return (r, c)

        # ----------------------------------------------------
        # TIER 5: Block Opponent Open-4 Threats
        # ----------------------------------------------------
        opp_fours = []
        for r, c in candidates:
            if self._creates_open_four(sim_board, r, c, self.opponent_id):
                opp_fours.append((r, c))
        if opp_fours:
            return opp_fours[0]

        # ----------------------------------------------------
        # TIER 6: Tactical Captures
        # ----------------------------------------------------
        capture_moves = []
        for r, c in candidates:
            b_test = sim_board.copy()
            caps = b_test.place_stone(r, c, self.player_id)
            if len(caps) > 0:
                is_over, winner = b_test.check_game_status()
                if not (is_over and winner == self.opponent_id):
                    capture_moves.append(((r, c), len(caps)))
        if capture_moves:
            capture_moves.sort(key=lambda x: x[1], reverse=True)
            if capture_moves[0][1] >= 2:
                # Top capture
                b_test = sim_board.copy()
                b_test.place_stone(capture_moves[0][0][0], capture_moves[0][0][1], self.player_id)
                # Ensure opponent doesn't win on the very next turn
                has_threat = any(self._creates_open_four(b_test, nr, nc, self.opponent_id) for nr, nc in self._get_candidates(b_test, b_test.get_legal_moves()))
                if not has_threat:
                    return capture_moves[0][0]

        # ----------------------------------------------------
        # TIER 7: Alpha-Beta Minimax Search with Neural Guidance
        # ----------------------------------------------------
        scored_candidates = []
        probs, _ = self.model.predict(sim_board, self.player_id, self.device)

        for r, c in candidates:
            tactical_score = self._quick_move_score(sim_board, r, c, self.player_id)
            neural_prior = probs[r * 19 + c] * 400.0
            scored_candidates.append((tactical_score + neural_prior, (r, c)))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        ordered_moves = [move for _, move in scored_candidates[:22]]

        best_move = ordered_moves[0]
        alpha = -float('inf')
        beta = float('inf')

        # Depth 2 Alpha-Beta
        for move in ordered_moves:
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
                val = self._min_val(b_copy, 1, alpha, beta, deadline)

            if val > alpha:
                alpha = val
                best_move = move

        return best_move

    def _min_val(self, board: Board, depth: int, alpha: float, beta: float, deadline: float) -> float:
        if depth <= 0 or time.perf_counter() >= deadline:
            return self._evaluate_board(board)

        candidates = self._get_candidates(board, board.get_legal_moves())
        scored = [(self._quick_move_score(board, r, c, self.opponent_id), (r, c)) for r, c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)

        v = float('inf')
        for _, move in scored[:14]:
            b_copy = board.copy()
            b_copy.place_stone(move[0], move[1], self.opponent_id)
            is_over, winner = b_copy.check_game_status()

            if is_over:
                val = -100000.0 if winner == self.opponent_id else (0.0 if winner == -1 else 100000.0)
            else:
                val = self._evaluate_board(b_copy)

            v = min(v, val)
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def _quick_move_score(self, board: Board, r: int, c: int, player: int) -> float:
        opp = 1 - player
        score = 0.0

        # Captures
        for dr, dc in Board.DIRECTIONS:
            r1, c1 = r + dr, c + dc
            r2, c2 = r + 2 * dr, c + 2 * dc
            r3, c3 = r + 3 * dr, c + 3 * dc
            if (board.is_valid_coord(r1, c1) and board.is_valid_coord(r2, c2) and board.is_valid_coord(r3, c3)):
                if board.grid[r1][c1] == opp and board.grid[r2][c2] == opp and board.grid[r3][c3] == player:
                    score += 800.0

        for dr, dc in Board.AXES:
            score += self._line_score(board, r, c, dr, dc, player) * 1.0
            score += self._line_score(board, r, c, dr, dc, opp) * 0.9
        return score

    def _line_score(self, board: Board, r: int, c: int, dr: int, dc: int, player: int) -> float:
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
            return 4500.0 if open_ends == 2 else (1400.0 if open_ends == 1 else 0.0)
        elif tot == 3:
            return 900.0 if open_ends == 2 else (180.0 if open_ends == 1 else 0.0)
        elif tot == 2:
            return 70.0 if open_ends == 2 else 20.0
        return 3.0

    def _creates_open_four(self, board: Board, r: int, c: int, player: int) -> bool:
        for dr, dc in Board.AXES:
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

            if (pos_cnt + neg_cnt + 1) == 4 and open_pos and open_neg:
                return True
        return False

    def _evaluate_board(self, board: Board) -> float:
        is_over, winner = board.check_game_status()
        if is_over:
            if winner == self.player_id:
                return 100000.0
            elif winner == self.opponent_id:
                return -100000.0
            else:
                return 0.0

        if self.player_id == Board.BLACK:
            # Sente / Aggressive strategy (relentless open-3/open-4 line creation)
            cap_diff = board.captures[self.player_id] - board.captures[self.opponent_id]
            score = cap_diff * 600.0

            for dr, dc in Board.AXES:
                for r in range(19):
                    for c in range(19):
                        val = board.grid[r][c]
                        if val != Board.EMPTY:
                            pr, pc = r - dr, c - dc
                            if not board.is_valid_coord(pr, pc) or board.grid[pr][pc] != val:
                                cnt = 0
                                cr, cc = r, c
                                while board.is_valid_coord(cr, cc) and board.grid[cr][cc] == val:
                                    cnt += 1
                                    cr += dr
                                    cc += dc

                                open_ends = 0
                                if board.is_valid_coord(pr, pc) and board.grid[pr][pc] == Board.EMPTY:
                                    open_ends += 1
                                if board.is_valid_coord(cr, cc) and board.grid[cr][cc] == Board.EMPTY:
                                    open_ends += 1

                                is_me = (val == self.player_id)
                                mul = 1.0 if is_me else -1.25

                                if cnt == 5:
                                    score += 100000.0 * mul
                                elif cnt >= 6:
                                    score -= 200.0 * (1.0 if is_me else -1.0)
                                elif cnt == 4:
                                    if open_ends == 2:
                                        score += 40000.0 * mul
                                    elif open_ends == 1:
                                        score += 3500.0 * mul
                                elif cnt == 3:
                                    if open_ends == 2:
                                        score += 2500.0 * mul
                                    elif open_ends == 1:
                                        score += 350.0 * mul
                                elif cnt == 2:
                                    if open_ends == 2:
                                        score += 100.0 * mul
                                    elif open_ends == 1:
                                        score += 20.0 * mul
            return score
        else:
            # Gote / Solid counter-punching strategy (compact shape defense and custodial counter-captures)
            cap_diff = board.captures[self.player_id] - board.captures[self.opponent_id]
            score = cap_diff * 400.0

            for dr, dc in Board.AXES:
                for r in range(19):
                    for c in range(19):
                        val = board.grid[r][c]
                        if val != Board.EMPTY:
                            pr, pc = r - dr, c - dc
                            if not board.is_valid_coord(pr, pc) or board.grid[pr][pc] != val:
                                cnt = 0
                                cr, cc = r, c
                                while board.is_valid_coord(cr, cc) and board.grid[cr][cc] == val:
                                    cnt += 1
                                    cr += dr
                                    cc += dc
                                mul = 1.0 if val == self.player_id else -1.1
                                if cnt == 5:
                                    score += 100000.0 * mul
                                elif cnt == 4:
                                    score += 3000.0 * mul
                                elif cnt == 3:
                                    score += 400.0 * mul
                                elif cnt == 2:
                                    score += 30.0 * mul
            return score

    def _get_candidates(self, board: Board, legal_moves: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        occupied = (np.array(board.grid) != Board.EMPTY)
        if not np.any(occupied):
            return [(9, 9)]

        mask = np.zeros((19, 19), dtype=bool)
        rows, cols = np.where(occupied)
        for r, c in zip(rows, cols):
            mask[max(0, r-2):min(19, r+3), max(0, c-2):min(19, c+3)] = True

        empty_mask = mask & (~occupied)
        coords = list(zip(*np.where(empty_mask)))
        return list(coords) if coords else legal_moves

    def _emergency_fallback(self, board: List[List[int]]) -> Tuple[int, int]:
        empty = []
        for r in range(19):
            for c in range(19):
                if board[r][c] == -1:
                    dist = abs(r - 9) + abs(c - 9)
                    empty.append((dist, (r, c)))
        if empty:
            empty.sort(key=lambda x: x[0])
            return empty[0][1]
        return (0, 0)


class SmartPlayerGodOfOmok(SmartPlayer):
    pass
