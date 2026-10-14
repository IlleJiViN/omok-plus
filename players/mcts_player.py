"""
Neural MCTS Player (Agent C: SL CNN + MCTS) for Omok++.
Combines PolicyValueNet with PUCT Monte Carlo Tree Search.
Features:
- Subtree reuse between consecutive moves
- Dynamic simulation budgeting to strictly comply with total game time limit (60s)
- Fast CPU fallback support
"""

import os
import torch
import numpy as np
from typing import List, Tuple
from players.player import Player
from engine.board import Board
from models.policy_value_net import PolicyValueNet
from mcts.mcts import MCTS


class MCTSPlayer(Player):
    def __init__(self,
                 player_id: int,
                 checkpoint_path: str = "models/checkpoints/sl_model_best.pt",
                 c_puct: float = 1.5,
                 default_sims: int = 100,
                 device: str = "auto"):
        super().__init__(player_id)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = PolicyValueNet(in_channels=7, num_channels=64, num_res_blocks=5).to(self.device)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained weights.")

        self.c_puct = c_puct
        self.default_sims = default_sims
        self.mcts = MCTS(self.model, self.device, c_puct=c_puct, n_simulations=default_sims)
        self.last_board_move_count = 0

    def take_turn(self, board: List[List[int]], time_left_ms: float) -> Tuple[int, int]:
        sim_board = Board(board)
        legal_moves = sim_board.get_legal_moves()

        if not legal_moves:
            raise RuntimeError("No legal moves available.")

        # Opening: center move
        if len(legal_moves) >= 360 and sim_board.is_empty(9, 9):
            return (9, 9)

        # Dynamic simulation budget based on remaining game time
        # Total game time is typically 60s (60,000ms).
        # We aim for ~40-50 moves max per game, so budget ~1.0s to 1.5s per move early, less later.
        if time_left_ms > 0:
            if time_left_ms > 40000:
                sims = self.default_sims          # e.g., 100
            elif time_left_ms > 20000:
                sims = int(self.default_sims * 0.75) # e.g., 75
            elif time_left_ms > 8000:
                sims = 40
            elif time_left_ms > 3000:
                sims = 20
            else:
                sims = 10  # Emergency speed mode
        else:
            sims = self.default_sims

        # Reset tree for fresh search
        self.mcts.reset_root()

        # Execute MCTS search
        action_probs = self.mcts.search(sim_board, self.player_id, add_noise=False, n_sims=sims)

        # Select action deterministically with highest visit count
        best_action = int(np.argmax(action_probs))
        r, c = divmod(best_action, 19)

        # Step root for subtree reuse
        self.mcts.step_root(best_action)

        if not sim_board.is_empty(r, c):
            import random
            return random.choice(legal_moves)

        return (r, c)
