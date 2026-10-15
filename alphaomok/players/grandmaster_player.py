"""
Grandmaster AlphaOmok Player:
The definitive tournament champion combining:
1. Mathematical VCF / VCT Proof Solver (Zero-Blunder Forced Mate Defense & Execution)
2. 7.88M Parameter ResNet-10 Neural Policy-Value Guidance (Deep Positional Intuition)
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
from friend_agent import AgentPlayer, to_rc, to_idx, BoardLightOfBucks


class GrandmasterPlayer(AgentPlayer):
    def __init__(self, color: int, weights_path: Optional[str] = None):
        super().__init__(color)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if weights_path is None:
            weights_path = os.path.join(base_dir, "models", "checkpoints", "rl_model_best.pt")

        self.net = PolicyValueNet.create_2x().to(self.device)
        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device)
            st = ckpt.get('model_state_dict') or ckpt
            self.net.load_state_dict(st)
        self.net.eval()
        self.sim_board = Board()

    def _neural_rank(self, candidates):
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        self.sim_board.grid = self.engine.to_external()
        probs, _ = self.net.predict(self.sim_board, self.color, self.device)
        return max(candidates, key=lambda idx: probs[to_rc(idx)[0] * 19 + to_rc(idx)[1]])

    def _tie_search(self, turn, ties):
        # Override tie search with pure ResNet-10 neural policy
        return self._neural_rank(ties)

    def _fallback_pick(self, turn, cands, l1_pass, label, nodes):
        # If there are surviving moves that don't lose immediately, let the neural net pick!
        st = self.stats
        ln = turn.ln
        l1 = turn.l1
        l2 = turn.l2

        if l1_pass:
            # Pick neural best among l1 passing moves
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
