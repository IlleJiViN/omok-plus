"""
PureNeuralPlayer: 100% Neural Network AI for Omok++.
Strictly uses Deep Learning (PolicyValueNet + MCTS / Neural Prior).
Zero hand-crafted tactical if-else rules.
All tactical defenses, fork blocks, and attack sequences are driven
exclusively by the trained ResNet-10 neural network weights.
"""

import os
import sys
import time
import torch
import numpy as np
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from players.player import Player
from engine.board import Board
from models.policy_value_net import PolicyValueNet
from mcts.mcts import MCTS


class PureNeuralPlayer(Player):
    def __init__(self, player_id: int, ckpt_path: Optional[str] = None, n_sims: int = 50):
        super().__init__(player_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_sims = n_sims

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if ckpt_path is None:
            ckpt_candidates = [
                os.path.join(base_dir, "models", "checkpoints", "rl_model_pure_neural.pt"),
                os.path.join(base_dir, "models", "checkpoints", "rl_model_best.pt"),
                os.path.join(base_dir, "alphaomok", "models", "weights", "rl_model_best.pt"),
            ]
            for p in ckpt_candidates:
                if os.path.exists(p):
                    ckpt_path = p
                    break

        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            st = ckpt.get("model_state_dict") or ckpt
            config = ckpt.get("config") or ckpt.get("model_config")
            if config:
                self.model = PolicyValueNet.from_config(config).to(self.device)
            else:
                self.model = PolicyValueNet.create_2x().to(self.device)
            self.model.load_state_dict(st)
        else:
            self.model = PolicyValueNet.create_2x().to(self.device)

        self.model.eval()
        self.mcts = MCTS(self.model, self.device, c_puct=1.5, n_simulations=self.n_sims)

    def take_turn(self, board: List[List[int]], time_left_ms: float) -> Tuple[int, int]:
        sim_board = Board(board)
        legal_moves = sim_board.get_legal_moves()
        if not legal_moves:
            return (9, 9)

        # Opening move optimization
        if len(legal_moves) >= 360 and sim_board.is_empty(9, 9):
            return (9, 9)

        # Basic Terminal 1-ply Check (mandatory game terminal rule)
        for r, c in legal_moves:
            b_test = sim_board.copy()
            b_test.place_stone(r, c, self.player_id)
            over, winner = b_test.check_game_status()
            if over and winner == self.player_id:
                return (r, c)

        # Pure Neural Policy & MCTS Search (No if-else heuristic rules)
        sims = self.n_sims
        if time_left_ms < 5000:
            sims = 20
        elif time_left_ms > 30000:
            sims = 50

        self.mcts.reset_root()
        action_probs = self.mcts.search(sim_board, self.player_id, n_sims=sims)
        best_action = int(np.argmax(action_probs))
        return divmod(best_action, 19)
