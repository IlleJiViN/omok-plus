"""
Comprehensive test script for EnhancedSmartPlayer vs FriendAgent.
Tests both Game 1 (AlphaOmok as Black) and Game 2 (AlphaOmok as White).
"""
import sys
import time
import importlib.util
from engine.board import Board

spec = importlib.util.spec_from_file_location('friend_agent', 'C:/Users/pc/Downloads/friend_agent.py')
fa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa)
FriendAgent = fa.AgentPlayer

from alphaomok.players.player import Player
from alphaomok.models.policy_value_net import PolicyValueNet
import torch
import numpy as np

class UltimateSmartPlayer(Player):
    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.opponent_id = 1 - player_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PolicyValueNet.create_baseline().to(self.device)
        self.model.eval()

    def _make_move(self, board: Board, r: int, c: int, player: int):
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

    def _undo_move(self, board: Board, r: int, c: int, player: int, caps):
        opp = 1 - player
        board.grid[r][c] = Board.EMPTY
        for cr, cc in caps:
            board.grid[cr][cc] = opp
        board.captures[player] -= len(caps)

    def _is_vulnerable(self, board: Board, r: int, c: int, player: int) -> bool:
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

    def _analyze_move_threats(self, board: Board, r: int, c: int, player: int) -> dict:
        caps = self._make_move(board, r, c, player)
        five_cnt = 0
        open_four_cnt = 0
        four_cnt = 0
        open_three_cnt = 0
        open_four_details = []

        for dr, dc in Board.AXES:
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
                continue

            axis_open_four = 0
            axis_four = 0

            if tot_P == 4:
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
                    open_four_details.append(((dr, dc), (nr1, nc1), (pr1, pc1)))
                elif neg_valid or pos_valid:
                    axis_four = 1
            else:
                broken_threats = set()
                for s in range(-4, 1):
                    vals = []
                    for k in range(s, s + 5):
                        rk, ck = r + k * dr, c + k * dc
                        if 0 <= rk < 19 and 0 <= ck < 19:
                            vals.append(board.grid[rk][ck])
                        else:
                            vals.append(-2)
                    if vals.count(player) == 4 and vals.count(Board.EMPTY) == 1:
                        rk_l, ck_l = r + (s - 1) * dr, c + (s - 1) * dc
                        b_left = board.grid[rk_l][ck_l] if (0 <= rk_l < 19 and 0 <= ck_l < 19) else -2
                        rk_r, ck_r = r + (s + 5) * dr, c + (s + 5) * dc
                        b_right = board.grid[rk_r][ck_r] if (0 <= rk_r < 19 and 0 <= ck_r < 19) else -2
                        if b_left != player and b_right != player:
                            broken_threats.add(s + vals.index(Board.EMPTY))
                axis_four = len(broken_threats)

            open_four_cnt += axis_open_four
            four_cnt += axis_four

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
            'vulnerable': vuln,
            'open_four_details': open_four_details
        }

    def take_turn(self, board, time_left_ms: float):
        sim_board = Board(board)
        legal_moves = sim_board.get_legal_moves()
        if not legal_moves:
            return (0, 0)

        # Opening: Black center move
        if len(legal_moves) >= 360 and sim_board.is_empty(9, 9):
            return (9, 9)

        # Opening: White response to (9, 9) -> Standard direct contact (8, 9)
        if len(legal_moves) == 360 and not sim_board.is_empty(9, 9):
            # Prefer (8, 9) or (9, 8)
            for pref in [(8, 9), (9, 8)]:
                if sim_board.is_empty(pref[0], pref[1]):
                    return pref

        start_time = time.perf_counter()
        turn_budget_ms = min(600.0, max(50.0, time_left_ms / 35.0)) if time_left_ms > 0 else 400.0
        deadline = start_time + (turn_budget_ms / 1000.0)

        candidates = self._get_candidates(sim_board, legal_moves)
        my_threats = {m: self._analyze_move_threats(sim_board, m[0], m[1], self.player_id) for m in candidates}
        opp_threats = {m: self._analyze_move_threats(sim_board, m[0], m[1], self.opponent_id) for m in candidates}

        # TIER 1: Immediate Winning Move (Exact 5 or 10 captures)
        for m in candidates:
            if my_threats[m]['five'] > 0 or (sim_board.captures[self.player_id] + my_threats[m]['captures'] >= 10):
                return (int(m[0]), int(m[1]))

        # TIER 2: Immediate Opponent Win Defense (Exact 5 or opponent 10 captures)
        opp_wins = [m for m in candidates if opp_threats[m]['five'] > 0 or (sim_board.captures[self.opponent_id] + opp_threats[m]['captures'] >= 10)]
        if opp_wins:
            # First check if we can safely block the win
            safe_wins = [m for m in opp_wins if not my_threats[m]['vulnerable']]
            if safe_wins:
                safe_wins.sort(key=lambda m: (my_threats[m]['captures'], my_threats[m]['four'], my_threats[m]['open_three']), reverse=True)
                return (int(safe_wins[0][0]), int(safe_wins[0][1]))

            # If all direct blocks are vulnerable, check if any capture move removes stones from opponent's winning line
            capturing_defenses = []
            for m in candidates:
                if my_threats[m]['captures'] >= 2 and not my_threats[m]['vulnerable']:
                    caps = self._make_move(sim_board, m[0], m[1], self.player_id)
                    rem_cands = [c for c in candidates if c != m]
                    rem_wins = sum(1 for c in rem_cands if self._analyze_move_threats(sim_board, c[0], c[1], self.opponent_id)['five'] > 0)
                    self._undo_move(sim_board, m[0], m[1], self.player_id, caps)
                    if rem_wins == 0:
                        capturing_defenses.append(m)

            if capturing_defenses:
                return (int(capturing_defenses[0][0]), int(capturing_defenses[0][1]))

            return (int(opp_wins[0][0]), int(opp_wins[0][1]))

        # TIER 3: Immediate Open-4 Creation (Guaranteed Win)
        my_open_fours = [m for m in candidates if my_threats[m]['open_four'] > 0]
        if my_open_fours:
            safe = [m for m in my_open_fours if not my_threats[m]['vulnerable']]
            pool = safe if safe else my_open_fours
            return (int(pool[0][0]), int(pool[0][1]))

        # TIER 4: Immediate Opponent Open-4 Block
        opp_open_fours = [m for m in candidates if opp_threats[m]['open_four'] > 0]
        if opp_open_fours:
            candidate_defenses = set()
            for m in opp_open_fours:
                if not my_threats[m]['vulnerable']:
                    candidate_defenses.add(m)
                for axis, end1, end2 in opp_threats[m].get('open_four_details', []):
                    for end in (end1, end2):
                        if sim_board.is_valid_coord(end[0], end[1]) and sim_board.is_empty(end[0], end[1]):
                            if not self._is_vulnerable(sim_board, end[0], end[1], self.player_id):
                                candidate_defenses.add(end)

            # Check if any safe capture actually eliminates the open four
            for m in candidates:
                if my_threats[m]['captures'] >= 2 and not my_threats[m]['vulnerable']:
                    caps = self._make_move(sim_board, m[0], m[1], self.player_id)
                    rem_cands = [c for c in candidates if c != m]
                    rem_of = sum(1 for c in rem_cands if self._analyze_move_threats(sim_board, c[0], c[1], self.opponent_id)['open_four'] > 0)
                    self._undo_move(sim_board, m[0], m[1], self.player_id, caps)
                    if rem_of < len(opp_open_fours):
                        candidate_defenses.add(m)

            defense_pool = list(candidate_defenses) if candidate_defenses else opp_open_fours
            if len(defense_pool) == 1:
                return (int(defense_pool[0][0]), int(defense_pool[0][1]))

            # Rank by strictly minimizing remaining open fours, then forks, then captures
            scored_defenses = []
            for m in defense_pool:
                caps = self._make_move(sim_board, m[0], m[1], self.player_id)
                rem_cands = [c for c in candidates if c != m]
                rem_opp = [self._analyze_move_threats(sim_board, c[0], c[1], self.opponent_id) for c in rem_cands]
                rem_fives = sum(1 for t in rem_opp if t['five'] > 0)
                rem_of = sum(1 for t in rem_opp if t['open_four'] > 0)
                rem_forks = sum(1 for t in rem_opp if (t['open_four'] + t['four'] >= 2) or (t['open_four'] + t['four'] >= 1 and t['open_three'] >= 1) or (t['open_three'] >= 2))
                rem_caps = sum(t['captures'] for t in rem_opp)
                self._undo_move(sim_board, m[0], m[1], self.player_id, caps)

                m_t = self._analyze_move_threats(sim_board, m[0], m[1], self.player_id)
                score = m_t['captures'] * 4000 + m_t['four'] * 800 + m_t['open_three'] * 400
                score -= rem_fives * 50000 + rem_of * 30000 + rem_forks * 4000 + rem_caps * 2500
                if m_t['vulnerable']:
                    score -= 20000
                score -= (abs(m[0] - 9) + abs(m[1] - 9)) * 10
                scored_defenses.append((score, m))

            scored_defenses.sort(key=lambda x: x[0], reverse=True)
            return (int(scored_defenses[0][1][0]), int(scored_defenses[0][1][1]))

        # TIER 5: Opponent Fork Defense & Own Fork Creation
        my_44 = [m for m in candidates if my_threats[m]['open_four'] + my_threats[m]['four'] >= 2]
        my_43 = [m for m in candidates if (my_threats[m]['open_four'] + my_threats[m]['four'] >= 1 and my_threats[m]['open_three'] >= 1)]
        my_33 = [m for m in candidates if my_threats[m]['open_three'] >= 2]

        opp_44 = [m for m in candidates if opp_threats[m]['open_four'] + opp_threats[m]['four'] >= 2]
        opp_43 = [m for m in candidates if (opp_threats[m]['open_four'] + opp_threats[m]['four'] >= 1 and opp_threats[m]['open_three'] >= 1)]
        opp_33 = [m for m in candidates if opp_threats[m]['open_three'] >= 2]

        # 1. Own 4-4 fork
        if my_44:
            safe = [m for m in my_44 if not my_threats[m]['vulnerable']]
            pool = safe if safe else my_44
            pool.sort(key=lambda m: (my_threats[m]['captures'], -abs(m[0]-9)-abs(m[1]-9)), reverse=True)
            return (int(pool[0][0]), int(pool[0][1]))

        # 2. Block Opponent 4-4 fork
        if opp_44:
            safe = [m for m in opp_44 if not my_threats[m]['vulnerable']]
            pool = safe if safe else opp_44
            pool.sort(key=lambda m: (my_threats[m]['captures'], my_threats[m]['four'], my_threats[m]['open_three'], -abs(m[0]-9)-abs(m[1]-9)), reverse=True)
            return (int(pool[0][0]), int(pool[0][1]))

        # 3. Own 4-3 fork
        if my_43:
            safe = [m for m in my_43 if not my_threats[m]['vulnerable']]
            pool = safe if safe else my_43
            pool.sort(key=lambda m: (my_threats[m]['captures'], -abs(m[0]-9)-abs(m[1]-9)), reverse=True)
            return (int(pool[0][0]), int(pool[0][1]))

        # 4. Block Opponent 4-3 fork
        if opp_43:
            safe = [m for m in opp_43 if not my_threats[m]['vulnerable']]
            pool = safe if safe else opp_43
            pool.sort(key=lambda m: (my_threats[m]['captures'], my_threats[m]['four'], my_threats[m]['open_three'], -abs(m[0]-9)-abs(m[1]-9)), reverse=True)
            return (int(pool[0][0]), int(pool[0][1]))

        # 5. Opponent 3-3 fork Defense
        if opp_33:
            safe = [m for m in opp_33 if not my_threats[m]['vulnerable']]
            pool = safe if safe else opp_33
            pool.sort(key=lambda m: (my_threats[m]['captures'], my_threats[m]['open_three'], -abs(m[0]-9)-abs(m[1]-9)), reverse=True)
            return (int(pool[0][0]), int(pool[0][1]))

        # 6. Own 3-3 fork
        if my_33:
            safe = [m for m in my_33 if not my_threats[m]['vulnerable']]
            pool = safe if safe else my_33
            pool.sort(key=lambda m: (my_threats[m]['captures'], -abs(m[0]-9)-abs(m[1]-9)), reverse=True)
            return (int(pool[0][0]), int(pool[0][1]))

        # TIER 6: Opponent Immediate Capture Defense
        opp_caps = [m for m in candidates if opp_threats[m]['captures'] >= 2]
        if opp_caps:
            safe_caps = [m for m in opp_caps if not my_threats[m]['vulnerable']]
            if safe_caps:
                safe_caps.sort(key=lambda m: (my_threats[m]['captures'], my_threats[m]['four'], my_threats[m]['open_three'], -abs(m[0]-9)-abs(m[1]-9)), reverse=True)
                return (int(safe_caps[0][0]), int(safe_caps[0][1]))

        # TIER 7: Own Safe Captures
        my_caps = [m for m in candidates if my_threats[m]['captures'] >= 2 and not my_threats[m]['vulnerable']]
        if my_caps:
            my_caps.sort(key=lambda m: (my_threats[m]['captures'], my_threats[m]['four'], my_threats[m]['open_three'], -abs(m[0]-9)-abs(m[1]-9)), reverse=True)
            return (int(my_caps[0][0]), int(my_caps[0][1]))

        # TIER 8, 9: Alpha-Beta Minimax Search (Depth 3)
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
                val = self._min_val(sim_board, 2, alpha, beta, deadline)
            self._undo_move(sim_board, move[0], move[1], self.player_id, caps)

            if val > alpha:
                alpha = val
                best_move = move

        return (int(best_move[0]), int(best_move[1]))

    def _quick_move_score(self, board: Board, r: int, c: int, player: int,
                          my_t: dict = None, opp_t: dict = None) -> float:
        opp = 1 - player
        if my_t is None:
            my_t = self._analyze_move_threats(board, r, c, player)
        if opp_t is None:
            opp_t = self._analyze_move_threats(board, r, c, opp)

        score = 0.0
        score += my_t['five'] * 100000.0
        score += opp_t['five'] * 80000.0
        score += my_t['open_four'] * 25000.0
        score += opp_t['open_four'] * 20000.0
        score += my_t['four'] * 4000.0
        score += opp_t['four'] * 3500.0

        my_fours = my_t['open_four'] + my_t['four']
        opp_fours = opp_t['open_four'] + opp_t['four']
        if my_fours >= 2:
            score += 18000.0
        elif my_fours >= 1 and my_t['open_three'] >= 1:
            score += 12000.0
        elif my_t['open_three'] >= 2:
            score += 6000.0

        if opp_fours >= 2:
            score += 15000.0
        elif opp_fours >= 1 and opp_t['open_three'] >= 1:
            score += 10000.0
        elif opp_t['open_three'] >= 2:
            score += 5000.0

        score += my_t['open_three'] * 1500.0
        score += opp_t['open_three'] * 1200.0

        score += my_t['captures'] * 4000.0
        score += opp_t['captures'] * 3800.0

        if my_t.get('vulnerable', False):
            score -= 20000.0

        dist = abs(r - 9) + abs(c - 9)
        score -= dist * 5.0
        return score

    def _min_val(self, board: Board, depth: int, alpha: float, beta: float, deadline: float) -> float:
        if depth <= 0 or time.perf_counter() >= deadline:
            return self._evaluate_board(board)

        candidates = self._get_candidates(board, board.get_legal_moves())
        scored = [(self._quick_move_score(board, r, c, self.opponent_id), (r, c)) for r, c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)

        v = float('inf')
        for _, move in scored[:8]:
            caps = self._make_move(board, move[0], move[1], self.opponent_id)
            is_over, winner = board.check_game_status()

            if is_over:
                val = -100000.0 if winner == self.opponent_id else (0.0 if winner == -1 else 100000.0)
            elif depth > 1:
                val = self._max_val(board, depth - 1, alpha, beta, deadline)
            else:
                val = self._evaluate_board(board)

            self._undo_move(board, move[0], move[1], self.opponent_id, caps)
            v = min(v, val)
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def _max_val(self, board: Board, depth: int, alpha: float, beta: float, deadline: float) -> float:
        if depth <= 0 or time.perf_counter() >= deadline:
            return self._evaluate_board(board)

        candidates = self._get_candidates(board, board.get_legal_moves())
        scored = [(self._quick_move_score(board, r, c, self.player_id), (r, c)) for r, c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)

        v = -float('inf')
        for _, move in scored[:8]:
            caps = self._make_move(board, move[0], move[1], self.player_id)
            is_over, winner = board.check_game_status()

            if is_over:
                val = 100000.0 if winner == self.player_id else (0.0 if winner == -1 else -100000.0)
            else:
                val = self._min_val(board, depth - 1, alpha, beta, deadline)

            self._undo_move(board, move[0], move[1], self.player_id, caps)
            v = max(v, val)
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

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
        score = cap_diff * 4000.0

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

    def _get_candidates(self, board: Board, legal_moves):
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


def play_full_game(black_player, white_player, max_turns=80):
    b = Board()
    curr = 0
    time_left = [60000.0, 60000.0]
    for t in range(1, max_turns + 1):
        is_over, winner = b.check_game_status()
        if is_over:
            return winner, t - 1
        p = black_player if curr == 0 else white_player
        t0 = time.perf_counter()
        mv = p.take_turn(b.get_grid(), time_left[curr])
        elapsed = (time.perf_counter() - t0) * 1000.0
        time_left[curr] -= elapsed
        caps = b.place_stone(mv[0], mv[1], curr)
        p_name = 'Black' if curr == 0 else 'White'
        cap_str = f' (caps: {len(caps)})' if caps else ''
        print(f'Turn {t:2d}: {p_name} -> ({mv[0]:2d}, {mv[1]:2d}){cap_str} (took {elapsed:.1f}ms, {time_left[curr]/1000:.1f}s left)')
        curr = 1 - curr
    is_over, winner = b.check_game_status()
    return (winner if is_over else -1), max_turns

if __name__ == '__main__':
    print('============================================================')
    print('GAME 1: UltimateSmartPlayer (Black) vs FriendAgent (White)')
    print('============================================================')
    p1 = UltimateSmartPlayer(0)
    p2 = FriendAgent(1)
    w1, t1 = play_full_game(p1, p2)
    print(f'Game 1 Result: Winner {w1} (0=Black, 1=White) at Turn {t1}\n')

    print('============================================================')
    print('GAME 2: FriendAgent (Black) vs UltimateSmartPlayer (White)')
    print('============================================================')
    p1 = FriendAgent(0)
    p2 = UltimateSmartPlayer(1)
    w2, t2 = play_full_game(p1, p2)
    print(f'Game 2 Result: Winner {w2} (0=Black, 1=White) at Turn {t2}')
