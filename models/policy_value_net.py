"""
PyTorch AlphaZero-style Policy-Value Network for Omok++.
Architecture:
- Input: 7-channel 19x19 tensor (Current player perspective)
  Ch 0: Current player's stones
  Ch 1: Opponent's stones
  Ch 2: Last move position
  Ch 3: Current player capture points (positions that capture opponent stones)
  Ch 4: Opponent capture points (positions where opponent can capture our stones)
  Ch 5: Color channel (1 if Black, 0 if White)
  Ch 6: Empty positions mask (1 if empty, 0 if occupied)
- Backbone: Lightweight ResNet (e.g., 5 ResBlocks, 64-96 channels)
- Policy Head: Conv1x1 -> BatchNorm -> ReLU -> Linear -> 361 logits
- Value Head: Conv1x1 -> BatchNorm -> ReLU -> Linear -> ReLU -> Linear -> Tanh [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from engine.board import Board


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class PolicyValueNet(nn.Module):
    def __init__(self,
                 in_channels: int = 7,
                 num_channels: int = 64,
                 num_res_blocks: int = 5,
                 policy_channels: int = 32,
                 value_channels: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.policy_channels = policy_channels
        self.value_channels = value_channels

        # Entry conv
        self.conv_in = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_channels)

        # Residual backbone
        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_res_blocks)])

        # Policy Head
        self.p_conv = nn.Conv2d(num_channels, policy_channels, kernel_size=1, bias=False)
        self.p_bn = nn.BatchNorm2d(policy_channels)
        self.p_fc = nn.Linear(policy_channels * 19 * 19, 361)

        # Value Head
        self.v_conv = nn.Conv2d(num_channels, value_channels, kernel_size=1, bias=False)
        self.v_bn = nn.BatchNorm2d(value_channels)
        self.v_fc1 = nn.Linear(value_channels * 19 * 19, 128)
        self.v_fc2 = nn.Linear(128, 1)

        self._init_weights()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_config(self) -> dict:
        """Returns the configuration dictionary to recreate this model architecture."""
        return {
            "in_channels": self.in_channels,
            "num_channels": self.num_channels,
            "num_res_blocks": self.num_res_blocks,
            "policy_channels": self.policy_channels,
            "value_channels": self.value_channels,
        }

    @classmethod
    def create_baseline(cls, in_channels: int = 7) -> "PolicyValueNet":
        """Factory method for baseline 5-block, 64-channel model."""
        return cls(
            in_channels=in_channels,
            num_channels=64,
            num_res_blocks=5,
            policy_channels=32,
            value_channels=16
        )

    @classmethod
    def create_2x(cls, in_channels: int = 7) -> "PolicyValueNet":
        """Factory method for 2x scaled 10-block, 128-channel model."""
        return cls(
            in_channels=in_channels,
            num_channels=128,
            num_res_blocks=10,
            policy_channels=32,
            value_channels=16
        )

    @classmethod
    def from_config(cls, config: dict) -> "PolicyValueNet":
        """Reconstruct model from configuration dictionary."""
        return cls(
            in_channels=config.get("in_channels", 7),
            num_channels=config.get("num_channels", 64),
            num_res_blocks=config.get("num_res_blocks", 5),
            policy_channels=config.get("policy_channels", 32),
            value_channels=config.get("value_channels", 16)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Small weights for final value and policy projections
        nn.init.uniform_(self.p_fc.weight, -0.01, 0.01)
        nn.init.uniform_(self.v_fc2.weight, -0.01, 0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, 7, 19, 19)
        Returns:
            policy_logits: (B, 361)
            value: (B, 1) in [-1, 1]
        """
        out = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = block(out)

        # Policy Head
        p = F.relu(self.p_bn(self.p_conv(out)))
        p = p.flatten(start_dim=1)
        policy_logits = self.p_fc(p)

        # Value Head
        v = F.relu(self.v_bn(self.v_conv(out)))
        v = v.flatten(start_dim=1)
        v = F.relu(self.v_fc1(v))
        value = torch.tanh(self.v_fc2(v))

        return policy_logits, value

    @staticmethod
    def encode_board(board: Board, current_player: int) -> np.ndarray:
        """
        Encodes Board state into a numpy array of shape (7, 19, 19) float32.
        Canonical representation: Channel 0 is always current player.
        """
        tensor = np.zeros((7, 19, 19), dtype=np.float32)
        opp = 1 - current_player

        # Ch 0: Current player's stones
        # Ch 1: Opponent's stones
        # Ch 6: Empty mask
        for r in range(19):
            for c in range(19):
                val = board.grid[r][c]
                if val == current_player:
                    tensor[0, r, c] = 1.0
                elif val == opp:
                    tensor[1, r, c] = 1.0
                else:
                    tensor[6, r, c] = 1.0

        # Ch 2: Last move
        if board.last_move is not None:
            lr, lc = board.last_move
            if 0 <= lr < 19 and 0 <= lc < 19:
                tensor[2, lr, lc] = 1.0

        # Ch 3: Current player capture positions
        # Ch 4: Opponent capture positions
        for r in range(19):
            for c in range(19):
                if board.grid[r][c] == Board.EMPTY:
                    # Check if playing here captures opponent
                    for dr, dc in Board.DIRECTIONS:
                        r1, c1 = r + dr, c + dc
                        r2, c2 = r + 2 * dr, c + 2 * dc
                        r3, c3 = r + 3 * dr, c + 3 * dc
                        if (board.is_valid_coord(r1, c1) and board.is_valid_coord(r2, c2) and board.is_valid_coord(r3, c3)):
                            if board.grid[r1][c1] == opp and board.grid[r2][c2] == opp and board.grid[r3][c3] == current_player:
                                tensor[3, r, c] = 1.0
                            if board.grid[r1][c1] == current_player and board.grid[r2][c2] == current_player and board.grid[r3][c3] == opp:
                                tensor[4, r, c] = 1.0

        # Ch 5: Color indicator
        if current_player == Board.BLACK:
            tensor[5, :, :] = 1.0

        return tensor

    def predict(self, board: Board, current_player: int, device: torch.device) -> Tuple[np.ndarray, float]:
        """
        Inference on a single board position.
        Returns:
            policy_probs: np.ndarray of shape (361,), illegal moves masked (prob=0).
            value: float in [-1, 1], evaluated from perspective of current_player.
        """
        self.eval()
        state_np = self.encode_board(board, current_player)
        state_t = torch.from_numpy(state_np).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, val = self.forward(state_t)
            logits = logits.squeeze(0).cpu().numpy()
            value = float(val.squeeze(0).cpu().item())

        # Mask illegal moves (already occupied cells)
        legal_mask = np.zeros(361, dtype=bool)
        for r in range(19):
            for c in range(19):
                if board.grid[r][c] == Board.EMPTY:
                    legal_mask[r * 19 + c] = True

        # Set illegal logits to very negative
        logits[~legal_mask] = -1e9

        # Numerically stable softmax over legal moves
        shift_logits = logits - np.max(logits[legal_mask]) if np.any(legal_mask) else logits
        exp_logits = np.exp(shift_logits)
        exp_logits[~legal_mask] = 0.0
        sum_exp = np.sum(exp_logits)

        if sum_exp > 0:
            probs = exp_logits / sum_exp
        else:
            probs = np.zeros(361, dtype=np.float32)
            if np.any(legal_mask):
                probs[legal_mask] = 1.0 / np.sum(legal_mask)

        return probs, value
