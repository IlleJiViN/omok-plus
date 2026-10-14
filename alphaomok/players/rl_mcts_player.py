"""
Self-Play Reinforcement Learning Player (Agent D: RL CNN + MCTS) for Omok++.
Combines the reinforced PolicyValueNet with PUCT Monte Carlo Tree Search.
"""

import os
import torch
import numpy as np
from typing import List, Tuple, Optional
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


class RLMCTSPlayer(Player):
    def __init__(self,
                 player_id: int,
                 checkpoint_path: Optional[str] = None,
                 c_puct: float = 1.5,
                 default_sims: int = 100,
                 device: str = "auto"):
        super().__init__(player_id)
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bundled_weights = os.path.join(base_dir, "models", "weights", "rl_model_best.pt")

        if checkpoint_path is None:
            checkpoint_path = bundled_weights
        elif not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(base_dir, checkpoint_path)

        # Fallback to SL model if RL model not yet saved
        if not os.path.exists(checkpoint_path):
            alt_path = os.path.join(base_dir, "models", "checkpoints", "rl_model_best.pt")
            if os.path.exists(alt_path):
                checkpoint_path = alt_path
            else:
                alt_path2 = os.path.join(base_dir, "models", "checkpoints", "sl_model_best.pt")
                if os.path.exists(alt_path2):
                    checkpoint_path = alt_path2

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            config = ckpt.get("config") or ckpt.get("model_config")
            if config:
                self.model = PolicyValueNet.from_config(config).to(self.device)
            elif "conv_in.weight" in state_dict:
                num_ch = state_dict["conv_in.weight"].shape[0]
                res_blocks = set(int(k.split(".")[1]) for k in state_dict.keys() if k.startswith("res_blocks."))
                num_res = len(res_blocks) if res_blocks else 5
                self.model = PolicyValueNet(num_channels=num_ch, num_res_blocks=num_res).to(self.device)
            else:
                self.model = PolicyValueNet.create_baseline().to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained weights.")
            self.model = PolicyValueNet.create_2x().to(self.device)

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

        # Time management for 60s game limit
        if time_left_ms > 0:
            if time_left_ms > 40000:
                sims = self.default_sims
            elif time_left_ms > 20000:
                sims = int(self.default_sims * 0.75)
            elif time_left_ms > 8000:
                sims = 40
            elif time_left_ms > 3000:
                sims = 20
            else:
                sims = 10
        else:
            sims = self.default_sims

        # Reset tree for fresh search
        self.mcts.reset_root()

        action_probs = self.mcts.search(sim_board, self.player_id, add_noise=False, n_sims=sims)
        best_action = int(np.argmax(action_probs))
        r, c = divmod(best_action, 19)

        self.mcts.step_root(best_action)

        if not sim_board.is_empty(r, c):
            import random
            return random.choice(legal_moves)

        return (r, c)
