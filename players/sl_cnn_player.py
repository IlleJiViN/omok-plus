"""
Supervised Learning CNN Player (Agent B) for Omok++.
Uses pure PolicyValueNet without tree search.
Features:
- Tactical win and block check (depth 1)
- Neural network policy prior over legal moves
- Fast inference (CPU/GPU)
"""

import os
import torch
import numpy as np
from typing import List, Tuple
from players.player import Player
from engine.board import Board
from models.policy_value_net import PolicyValueNet


class SLCNNPlayer(Player):
    def __init__(self, player_id: int, checkpoint_path: str = "models/checkpoints/sl_model_best.pt", device: str = "auto"):
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

    def take_turn(self, board: List[List[int]], time_left_ms: float) -> Tuple[int, int]:
        sim_board = Board(board)
        legal_moves = sim_board.get_legal_moves()

        if not legal_moves:
            raise RuntimeError("No legal moves available.")

        # Opening: center on empty board
        if len(legal_moves) >= 360 and sim_board.is_empty(9, 9):
            return (9, 9)

        # 1. Immediate Win
        for r, c in legal_moves:
            b_copy = sim_board.copy()
            b_copy.place_stone(r, c, self.player_id)
            is_over, winner = b_copy.check_game_status()
            if is_over and winner == self.player_id:
                return (r, c)

        # 2. Block Opponent Win
        opp = 1 - self.player_id
        opp_wins = []
        for r, c in legal_moves:
            b_copy = sim_board.copy()
            b_copy.place_stone(r, c, opp)
            is_over, winner = b_copy.check_game_status()
            if is_over and winner == opp:
                opp_wins.append((r, c))
        if opp_wins:
            return opp_wins[0]

        # 3. Policy Network Inference
        probs, value = self.model.predict(sim_board, self.player_id, self.device)

        # Pick highest probability legal move
        best_move_idx = int(np.argmax(probs))
        r, c = divmod(best_move_idx, 19)

        if not sim_board.is_empty(r, c):
            # Fallback to random legal move if masked index fails
            import random
            return random.choice(legal_moves)

        return (r, c)
