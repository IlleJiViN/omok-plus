"""
Grandmaster AlphaOmok Player:
The definitive tournament champion combining:
1. Mathematical VCF / VCT Proof Solver (Zero-Blunder Forced Mate Defense & Execution)
2. 7.88M Parameter ResNet-10 Neural Policy-Value Guidance (Deep Positional Intuition)
3. Custodial Capture & Sandwich Trap Tactical Awareness
4. Neural Opening Director (_book_order override against hardcoded opponent book traps)
"""

import os
import sys
import torch
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'C:/Users/pc/Downloads')

from engine.board import Board
from models.policy_value_net import PolicyValueNet
from friend_agent import AgentPlayer, to_rc, to_idx, BoardLightOfBucks, _TurnLightOfBucks


class GrandmasterPlayer(AgentPlayer):
    def __init__(self, color: int, weights_path: Optional[str] = None):
        super().__init__(color)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if weights_path is None:
            # Check package weights first, then models/checkpoints
            pkg_weights = os.path.join(base_dir, "alphaomok", "models", "weights", "rl_model_best.pt")
            ckpt_weights = os.path.join(base_dir, "models", "checkpoints", "rl_model_best.pt")
            weights_path = pkg_weights if os.path.exists(pkg_weights) else ckpt_weights

        self.net = PolicyValueNet.create_2x().to(self.device)
        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device)
            st = ckpt.get('model_state_dict') or ckpt
            self.net.load_state_dict(st)
        self.net.eval()
        self.sim_board = Board()

    def _score_candidate(self, idx: int, probs: np.ndarray) -> float:
        r, c = to_rc(idx)
        base = float(probs[r * 19 + c])
        
        # Tactical Bonus: Active custodial capture
        caps = self.engine.capture_cells(idx, self.color)
        if caps:
            base += len(caps) * 0.35
            
        # Tactical Penalty: Moving into an immediate custodial capture trap (X-O-O-X sandwich)
        opp = 1 - self.color
        for d in (1, -1, 23, -23, 24, -24, 22, -22):
            adj = idx + d
            if self.engine.cells[adj] == self.color:
                front = adj + d
                back = idx - d
                if ((self.engine.cells[back] == opp and self.engine.cells[front] == 2) or
                    (self.engine.cells[front] == opp and self.engine.cells[back] == 2)):
                    base -= 0.6
                    break
        return base

    def _neural_rank(self, candidates):
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        self.sim_board.grid = self.engine.to_external()
        probs, _ = self.net.predict(self.sim_board, self.color, self.device)
        return max(candidates, key=lambda idx: self._score_candidate(idx, probs))

    def _book_order(self, res, moves, out):
        """Neural Opening Director: Prioritize opening moves according to ResNet-10 policy."""
        if not moves:
            return None
        self.sim_board.grid = self.engine.to_external()
        probs, _ = self.net.predict(self.sim_board, self.color, self.device)
        best = max(moves, key=lambda idx: probs[to_rc(idx)[0] * 19 + to_rc(idx)[1]])
        out.append(best)
        for m in moves:
            if m != best:
                out.append(m)
        self.stats['last_book'] = 'neural_hit'
        return out

    def _choose_safe(self, moves, t0, budget, hard, time=None):
        orig_deep_pick = _TurnLightOfBucks.deep_pick
        player_self = self

        def enhanced_deep_pick(turn_self, safe):
            cands = safe[:10]
            if len(cands) < 2:
                return cands[0]
            orig_best = orig_deep_pick(turn_self, safe)
            max_q = turn_self.quality(orig_best)
            top_q = [m for m in cands if turn_self.quality(m) >= max_q]
            if not top_q:
                top_q = cands
            return player_self._neural_rank(top_q)

        _TurnLightOfBucks.deep_pick = enhanced_deep_pick
        try:
            return super()._choose_safe(moves, t0, budget, hard, time)
        finally:
            _TurnLightOfBucks.deep_pick = orig_deep_pick

    def _tie_search(self, turn, ties):
        return self._neural_rank(ties)

    def _fallback_pick(self, turn, cands, l1_pass, label, nodes):
        st = self.stats
        ln = turn.ln
        l1 = turn.l1
        l2 = turn.l2

        if l1_pass:
            pick = self._neural_rank(l1_pass)
            st['last_vcf'] = label
            if pick in l2:
                st['last_vct'] = 'unknown' if l2[pick] is None else 'lost'
            return self._done(pick, turn, nodes)

        alive = [m for m in cands if ln.get(m) is False]
        if alive:
            pick = self._neural_rank(alive)
        elif cands:
            pick = self._neural_rank(cands)
        else:
            pick = None

        st['last_vcf'] = 'lost' if label == 'break' else 'checked'
        return self._done(pick, turn, nodes)
