"""
SmartPlayer: High-Performance Tactical AI for Omok++.

Comprehensive Tactical Decision Hierarchy:
- TIER 1: Immediate Winning Move (Exact 5-in-a-row) (0ms)
- TIER 2: Immediate Opponent Win Defense (Exact 5) (0ms)
- TIER 3: Immediate Open-4 Creation (Guaranteed Win)
- TIER 4: Immediate Opponent Open-4 Block (Threat-minimizing defense)
- TIER 5: Opponent Fork Defense (Double-Threat Block: 4-4, 4-3, 3-3 forks)
- TIER 6: Own Fork Creation (4-4, 4-3, 3-3 forks)
- TIER 7: Opponent Four Block (Broken Four / Closed Four Defense vs Counter-Threat)
- TIER 8: Tactical Captures (Safe capture execution)
- TIER 9: Alpha-Beta Minimax with ResNet-10 Neural Guidance
"""

import os
import sys
import time
import random
import torch
import numpy as np
from typing import List, Tuple, Optional, Set, Dict

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

    # ----------------------------------------------------------------------
    # Fast In-Place Move Execution and Undo (Zero Board Copying)
    # ----------------------------------------------------------------------
    def _make_move(self, board: Board, r: int, c: int, player: int) -> List[Tuple[int, int]]:
        opp = 1 - player
        board.grid[r][c] = player
        caps = []
        for dr, dc in Board.DIRECTIONS:
            r1, c1 = r + dr, c + dc
            r2, c2 = r + 2 * dr, c + 2 * dc
            r3, c3 = r + 3 * dr, c + 3 * dc
            if (0 <= r3 < 19 and 0 <= c3 < 19 and
                board.grid[r1][c1] == opp and
                board.grid[r2][c2] == opp and
                board.grid[r3][c3] == player):
                caps.append((r1, c1))
                caps.append((r2, c2))
        for cr, cc in caps:
            board.grid[cr][cc] = Board.EMPTY
        board.captures[player] += len(caps)
        return caps

    def _undo_move(self, board: Board, r: int, c: int, player: int, caps: List[Tuple[int, int]]):
        opp = 1 - player
        board.grid[r][c] = Board.EMPTY
        for cr, cc in caps:
            board.grid[cr][cc] = opp
        board.captures[player] -= len(caps)

    def _is_vulnerable(self, board: Board, r: int, c: int, player: int) -> bool:
        """Checks if placing player at (r, c) creates a pair that opponent can immediately capture."""
        opp = 1 - player
        board.grid[r][c] = player
        vuln = False
        for dr, dc in Board.DIRECTIONS:
            r1, c1 = r + dr, c + dc
            if 0 <= r1 < 19 and 0 <= c1 < 19 and board.grid[r1][c1] == player:
                r_back, c_back = r - dr, c - dc
                r_front, c_front = r1 + dr, c1 + dc
                if (0 <= r_back < 19 and 0 <= c_back < 19 and board.grid[r_back][c_back] == opp and
                    0 <= r_front < 19 and 0 <= c_front < 19 and board.grid[r_front][c_front] == Board.EMPTY):
                    vuln = True
                    break
                if (0 <= r_front < 19 and 0 <= c_front < 19 and board.grid[r_front][c_front] == opp and
                    0 <= r_back < 19 and 0 <= c_back < 19 and board.grid[r_back][c_back] == Board.EMPTY):
                    vuln = True
                    break
        board.grid[r][c] = Board.EMPTY
        return vuln

    # ----------------------------------------------------------------------
    # Comprehensive Pattern and Threat Analysis along 4 Axes
    # ----------------------------------------------------------------------
    def _analyze_move_threats(self, board: Board, r: int, c: int, player: int) -> dict:
        """
        Analyzes threats created if `player` plays at `(r, c)`.
        Simulates move, executes captures, checks all 4 line axes for:
        - five: exact 5-in-a-row (respecting 6+ overline rule)
        - open_four: contiguous 4 with both ends open (neither overlines)
        - four: single-ended or broken fours (1 move away from exact 5)
        - open_three: open 3s capable of becoming an open 4
        - captures: number of stones captured
        - vulnerable: whether move creates an immediately capturable pair
        """
        caps = self._make_move(board, r, c, player)
        five_cnt = 0
        open_four_cnt = 0
        four_cnt = 0
        open_three_cnt = 0

        for dr, dc in Board.AXES:
            # 1. Contiguous run check
            pos_cnt = 0
            step = 1
            while 0 <= r + step * dr < 19 and 0 <= c + step * dc < 19 and board.grid[r + step * dr][c + step * dc] == player:
                pos_cnt += 1
                step += 1
            neg_cnt = 0
            step = 1
            while 0 <= r - step * dr < 19 and 0 <= c - step * dc < 19 and board.grid[r - step * dr][c - step * dc] == player:
                neg_cnt += 1
                step += 1
            tot_P = pos_cnt + neg_cnt + 1

            if tot_P == 5:
                five_cnt += 1
                continue
            elif tot_P > 5:
                # Overline in Omok++: does not count as five or four
                continue

            axis_open_four = 0
            axis_four = 0

            if tot_P == 4:
                # Contiguous 4: check both ends
                nr1, nc1 = r - (neg_cnt + 1) * dr, c - (neg_cnt + 1) * dc
                nr2, nc2 = r - (neg_cnt + 2) * dr, c - (neg_cnt + 2) * dc
                neg_valid = (0 <= nr1 < 19 and 0 <= nc1 < 19 and board.grid[nr1][nc1] == Board.EMPTY and
                             (not (0 <= nr2 < 19 and 0 <= nc2 < 19) or board.grid[nr2][nc2] != player))

                pr1, pc1 = r + (pos_cnt + 1) * dr, c + (pos_cnt + 1) * dc
                pr2, pc2 = r + (pos_cnt + 2) * dr, c + (pos_cnt + 2) * dc
                pos_valid = (0 <= pr1 < 19 and 0 <= pc1 < 19 and board.grid[pr1][pc1] == Board.EMPTY and
                             (not (0 <= pr2 < 19 and 0 <= pc2 < 19) or board.grid[pr2][pc2] != player))

                if neg_valid and pos_valid:
                    axis_open_four = 1
                elif neg_valid or pos_valid:
                    axis_four = 1
            else:
                # Broken fours: 5-cell window check containing (r, c) (s in [-4, 0])
                broken_threats = set()
                for s in range(-4, 1):
                    vals = []
                    for k in range(s, s + 5):
                        rk, ck = r + k * dr, c + k * dc
                        if 0 <= rk < 19 and 0 <= ck < 19:
                            vals.append(board.grid[rk][ck])
                        else:
                            vals.append(-2)  # Wall / off-board
                    if vals.count(player) == 4 and vals.count(Board.EMPTY) == 1:
                        # Check boundaries to ensure completion does not overline
                        rk_l, ck_l = r + (s - 1) * dr, c + (s - 1) * dc
                        b_left = board.grid[rk_l][ck_l] if (0 <= rk_l < 19 and 0 <= ck_l < 19) else -2
                        rk_r, ck_r = r + (s + 5) * dr, c + (s + 5) * dc
                        b_right = board.grid[rk_r][ck_r] if (0 <= rk_r < 19 and 0 <= ck_r < 19) else -2
                        if b_left != player and b_right != player:
                            broken_threats.add(s + vals.index(Board.EMPTY))
                axis_four = len(broken_threats)

            open_four_cnt += axis_open_four
            four_cnt += axis_four

            # Check open three only if this axis does not already have a four or open four
            if axis_open_four == 0 and axis_four == 0:
                for s in range(-4, 0):
                    rk_s, ck_s = r + s * dr, c + s * dc
                    rk_e, ck_e = r + (s + 5) * dr, c + (s + 5) * dc
                    if (0 <= rk_s < 19 and 0 <= ck_s < 19 and board.grid[rk_s][ck_s] == Board.EMPTY and
                        0 <= rk_e < 19 and 0 <= ck_e < 19 and board.grid[rk_e][ck_e] == Board.EMPTY):
                        mid_p = 0
                        mid_e = 0
                        other = False
                        for k in range(s + 1, s + 5):
                            v = board.grid[r + k * dr][c + k * dc]
                            if v == player:
                                mid_p += 1
                            elif v == Board.EMPTY:
                                mid_e += 1
                            else:
                                other = True
                                break
                        if not other and mid_p == 3 and mid_e == 1:
                            rk_l, ck_l = r + (s - 1) * dr, c + (s - 1) * dc
                            b_left = board.grid[rk_l][ck_l] if (0 <= rk_l < 19 and 0 <= ck_l < 19) else -2
                            rk_r, ck_r = r + (s + 6) * dr, c + (s + 6) * dc
                            b_right = board.grid[rk_r][ck_r] if (0 <= rk_r < 19 and 0 <= ck_r < 19) else -2
                            if b_left != player and b_right != player:
                                open_three_cnt += 1
                                break

        self._undo_move(board, r, c, player, caps)
        vuln = self._is_vulnerable(board, r, c, player)
        return {
            'five': five_cnt,
            'open_four': open_four_cnt,
            'four': four_cnt,
            'open_three': open_three_cnt,
            'captures': len(caps),
            'vulnerable': vuln
        }

    # ----------------------------------------------------------------------
    # Tactical Decision Pipeline
    # ----------------------------------------------------------------------
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

        # Candidate Generation (Chebyshev radius 2 around occupied cells)
        candidates = self._get_candidates(sim_board, legal_moves)

        # Precompute Threat Analysis across candidates (< 2ms total)
        my_threats = {m: self._analyze_move_threats(sim_board, m[0], m[1], self.player_id) for m in candidates}
        opp_threats = {m: self._analyze_move_threats(sim_board, m[0], m[1], self.opponent_id) for m in candidates}

        # ----------------------------------------------------
        # TIER 1: Immediate Winning Move (Exact 5-in-a-row) (0ms)
        # ----------------------------------------------------
        for m in candidates:
            if my_threats[m]['five'] > 0:
                return (int(m[0]), int(m[1]))

        # ----------------------------------------------------
        # TIER 2: Immediate Opponent Win Defense (Exact 5) (0ms)
        # ----------------------------------------------------
        opp_wins = [m for m in candidates if opp_threats[m]['five'] > 0]
        if opp_wins:
            if len(opp_wins) == 1:
                return (int(opp_wins[0][0]), int(opp_wins[0][1]))
            # Opponent has multiple winning moves: pick block that is not vulnerable or captures most
            safe_wins = [m for m in opp_wins if not my_threats[m]['vulnerable']]
            pool = safe_wins if safe_wins else opp_wins
            pool.sort(key=lambda m: (my_threats[m]['captures'], my_threats[m]['four'], my_threats[m]['open_three']), reverse=True)
            return (int(pool[0][0]), int(pool[0][1]))

        # ----------------------------------------------------
        # TIER 3: Immediate Open-4 Creation (Guaranteed Win)
        # ----------------------------------------------------
        my_open_fours = [m for m in candidates if my_threats[m]['open_four'] > 0]
        if my_open_fours:
            return (int(my_open_fours[0][0]), int(my_open_fours[0][1]))

        # ----------------------------------------------------
        # TIER 4: Immediate Opponent Open-4 Block
        # ----------------------------------------------------
        opp_open_fours = [m for m in candidates if opp_threats[m]['open_four'] > 0]
        if opp_open_fours:
            # Filter out vulnerable blocks that get captured next turn
            safe_blocks = [m for m in opp_open_fours if not my_threats[m]['vulnerable']]
            block_pool = safe_blocks if safe_blocks else opp_open_fours
            if len(block_pool) == 1:
                return (int(block_pool[0][0]), int(block_pool[0][1]))

            # Multiple blocks: evaluate which block best suppresses opponent's remaining threats
            best_block = block_pool[0]
            best_score = -float('inf')
            for m in block_pool:
                caps = self._make_move(sim_board, m[0], m[1], self.player_id)
                rem_cands = [c for c in candidates if c != m]
                rem_opp_threats = [self._analyze_move_threats(sim_board, c[0], c[1], self.opponent_id) for c in rem_cands]
                rem_forks = sum(1 for t in rem_opp_threats if (t['open_four'] + t['four'] >= 2) or (t['open_four'] + t['four'] >= 1 and t['open_three'] >= 1) or (t['open_three'] >= 2))
                rem_fours = sum(1 for t in rem_opp_threats if t['open_four'] + t['four'] >= 1)
                rem_caps = sum(t['captures'] for t in rem_opp_threats)
                self._undo_move(sim_board, m[0], m[1], self.player_id, caps)

                score = my_threats[m]['captures'] * 3500 + my_threats[m]['four'] * 800 + my_threats[m]['open_three'] * 400
                score -= rem_forks * 4000 + rem_fours * 600 + rem_caps * 3000
                score -= (abs(m[0] - 9) + abs(m[1] - 9)) * 10
                if score > best_score:
                    best_score = score
                    best_block = m
            return (int(best_block[0]), int(best_block[1]))

        # ----------------------------------------------------
        # TIER 5 & 6: Tactical Fork Analysis (4-4, 4-3, 3-3)
        # ----------------------------------------------------
        my_44 = [m for m in candidates if my_threats[m]['open_four'] + my_threats[m]['four'] >= 2]
        my_43 = [m for m in candidates if (my_threats[m]['open_four'] + my_threats[m]['four'] >= 1 and my_threats[m]['open_three'] >= 1)]
        my_33 = [m for m in candidates if my_threats[m]['open_three'] >= 2]

        opp_44 = [m for m in candidates if opp_threats[m]['open_four'] + opp_threats[m]['four'] >= 2]
        opp_43 = [m for m in candidates if (opp_threats[m]['open_four'] + opp_threats[m]['four'] >= 1 and opp_threats[m]['open_three'] >= 1)]
        opp_33 = [m for m in candidates if opp_threats[m]['open_three'] >= 2]

        # 1. Own 4-4 fork: Guaranteed win in 2 moves
        if my_44:
            safe_my_44 = [m for m in my_44 if not my_threats[m]['vulnerable']]
            pool = safe_my_44 if safe_my_44 else my_44
            pool.sort(key=lambda m: (my_threats[m]['captures'], -abs(m[0]-9)-abs(m[1]-9)), reverse=True)
            return (int(pool[0][0]), int(pool[0][1]))

        # 2. Block Opponent 4-4 fork: Must block fork vertex!
        if opp_44:
            safe_opp_44 = [m for m in opp_44 if not my_threats[m]['vulnerable']]
            pool = safe_opp_44 if safe_opp_44 else opp_44
            pool.sort(key=lambda m: (my_threats[m]['captures'], my_threats[m]['four'], my_threats[m]['open_three'], -abs(m[0]-9)-abs(m[1]-9)), reverse=True)
            return (int(pool[0][0]), int(pool[0][1]))

        # 3. Own 4-3 fork: Guaranteed win in 3 moves
        if my_43:
            safe_my_43 = [m for m in my_43 if not my_threats[m]['vulnerable']]
            pool = safe_my_43 if safe_my_43 else my_43
            pool.sort(key=lambda m: (my_threats[m]['captures'], -abs(m[0]-9)-abs(m[1]-9)), reverse=True)
            return (int(pool[0][0]), int(pool[0][1]))

        # 4. Block Opponent 4-3 fork: Must block fork vertex!
        if opp_43:
            safe_opp_43 = [m for m in opp_43 if not my_threats[m]['vulnerable']]
            pool = safe_opp_43 if safe_opp_43 else opp_43
            pool.sort(key=lambda m: (my_threats[m]['captures'], my_threats[m]['four'], my_threats[m]['open_three'], -abs(m[0]-9)-abs(m[1]-9)), reverse=True)
            return (int(pool[0][0]), int(pool[0][1]))

        # 5. Opponent 3-3 fork Defense:
        if opp_33:
            fours_with_sente = [m for m in candidates if my_threats[m]['four'] >= 1 and not my_threats[m]['vulnerable']]
            if fours_with_sente:
                fours_with_sente.sort(key=lambda m: (my_threats[m]['captures'], my_threats[m]['open_three'], -abs(m[0]-9)-abs(m[1]-9)), reverse=True)
                return (int(fours_with_sente[0][0]), int(fours_with_sente[0][1]))
            safe_opp_33 = [m for m in opp_33 if not my_threats[m]['vulnerable']]
            pool = safe_opp_33 if safe_opp_33 else opp_33
            pool.sort(key=lambda m: (my_threats[m]['captures'], my_threats[m]['open_three'], -abs(m[0]-9)-abs(m[1]-9)), reverse=True)
            return (int(pool[0][0]), int(pool[0][1]))

        # 6. Own 3-3 fork:
        if my_33:
            safe_my_33 = [m for m in my_33 if not my_threats[m]['vulnerable']]
            pool = safe_my_33 if safe_my_33 else my_33
            pool.sort(key=lambda m: (my_threats[m]['captures'], -abs(m[0]-9)-abs(m[1]-9)), reverse=True)
            return (int(pool[0][0]), int(pool[0][1]))

        # ----------------------------------------------------
        # TIER 7, 8, 9: Tactical Foresight & Alpha-Beta Minimax
        # Evaluates candidate moves considering opponent counter-threats,
        # captures, defenses, and ResNet-10 neural priors.
        # ----------------------------------------------------
        scored_candidates = []
        probs, _ = self.model.predict(sim_board, self.player_id, self.device)

        for r, c in candidates:
            tactical_score = self._quick_move_score(sim_board, r, c, self.player_id, my_threats[(r, c)], opp_threats[(r, c)])
            neural_prior = probs[r * 19 + c] * 400.0
            scored_candidates.append((tactical_score + neural_prior, (r, c)))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        ordered_moves = [move for _, move in scored_candidates[:16]]

        best_move = ordered_moves[0]
        alpha = -float('inf')
        beta = float('inf')

        # Depth 2 Alpha-Beta Search with Tactical Counter-Evaluation
        for move in ordered_moves:
            if time.perf_counter() >= deadline:
                break

            caps = self._make_move(sim_board, move[0], move[1], self.player_id)
            is_over, winner = sim_board.check_game_status()

            if is_over:
                if winner == self.player_id:
                    self._undo_move(sim_board, move[0], move[1], self.player_id, caps)
                    return (int(move[0]), int(move[1]))
                elif winner == -1:
                    val = 0.0
                else:
                    val = -100000.0
            else:
                val = self._min_val(sim_board, 1, alpha, beta, deadline)

            self._undo_move(sim_board, move[0], move[1], self.player_id, caps)

            if val > alpha:
                alpha = val
                best_move = move

        return (int(best_move[0]), int(best_move[1]))

    def _min_val(self, board: Board, depth: int, alpha: float, beta: float, deadline: float) -> float:
        if depth <= 0 or time.perf_counter() >= deadline:
            return self._evaluate_board(board)

        candidates = self._get_candidates(board, board.get_legal_moves())
        scored = [(self._quick_move_score(board, r, c, self.opponent_id), (r, c)) for r, c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)

        v = float('inf')
        for _, move in scored[:12]:
            caps = self._make_move(board, move[0], move[1], self.opponent_id)
            is_over, winner = board.check_game_status()

            if is_over:
                val = -100000.0 if winner == self.opponent_id else (0.0 if winner == -1 else 100000.0)
            else:
                val = self._evaluate_board(board)

            self._undo_move(board, move[0], move[1], self.opponent_id, caps)

            v = min(v, val)
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def _quick_move_score(self, board: Board, r: int, c: int, player: int,
                          my_t: Optional[dict] = None, opp_t: Optional[dict] = None) -> float:
        opp = 1 - player
        if my_t is None:
            my_t = self._analyze_move_threats(board, r, c, player)
        if opp_t is None:
            opp_t = self._analyze_move_threats(board, r, c, opp)

        score = 0.0
        # Five
        score += my_t['five'] * 100000.0
        score += opp_t['five'] * 80000.0

        # Open four
        score += my_t['open_four'] * 25000.0
        score += opp_t['open_four'] * 20000.0

        # Fours (broken or closed)
        score += my_t['four'] * 4000.0
        score += opp_t['four'] * 3500.0

        # Forks bonus
        my_fours = my_t['open_four'] + my_t['four']
        opp_fours = opp_t['open_four'] + opp_t['four']
        if my_fours >= 2:
            score += 18000.0  # 4-4 fork
        elif my_fours >= 1 and my_t['open_three'] >= 1:
            score += 12000.0  # 4-3 fork
        elif my_t['open_three'] >= 2:
            score += 6000.0   # 3-3 fork

        if opp_fours >= 2:
            score += 15000.0  # block opp 4-4 fork
        elif opp_fours >= 1 and opp_t['open_three'] >= 1:
            score += 10000.0  # block opp 4-3 fork
        elif opp_t['open_three'] >= 2:
            score += 5000.0   # block opp 3-3 fork

        # Open three
        score += my_t['open_three'] * 1500.0
        score += opp_t['open_three'] * 1200.0

        # Captures
        score += my_t['captures'] * 3500.0
        score += opp_t['captures'] * 3000.0  # defending against opponent capture

        # Vulnerability penalty: severely penalize moving into an immediate opponent capture
        if my_t.get('vulnerable', False):
            score -= 10000.0

        # Positional center bias
        dist_from_center = abs(r - 9) + abs(c - 9)
        score -= dist_from_center * 5.0

        return score

    def _evaluate_board(self, board: Board) -> float:
        is_over, winner = board.check_game_status()
        if is_over:
            if winner == self.player_id:
                return 100000.0
            elif winner == self.opponent_id:
                return -100000.0
            else:
                return 0.0

        cap_diff = board.captures[self.player_id] - board.captures[self.opponent_id]
        score = cap_diff * 3500.0

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
                            mul = 1.0 if is_me else -1.2

                            if cnt == 5:
                                score += 100000.0 * mul
                            elif cnt >= 6:
                                score -= 200.0 * (1.0 if is_me else -1.0)
                            elif cnt == 4:
                                if open_ends == 2:
                                    score += 30000.0 * mul
                                elif open_ends == 1:
                                    score += 3500.0 * mul
                            elif cnt == 3:
                                if open_ends == 2:
                                    score += 2500.0 * mul
                                elif open_ends == 1:
                                    score += 400.0 * mul
                            elif cnt == 2:
                                if open_ends == 2:
                                    score += 100.0 * mul
                                elif open_ends == 1:
                                    score += 20.0 * mul
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
