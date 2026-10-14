"""
AlphaOmok High-Level Public API.
Provides:
- AnalysisResult: Evaluation dataclass
- AlphaOmok: AI player and board analyzer
- Game: High-level interactive match wrapper
"""

import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

import numpy as np
import torch

from alphaomok.engine.board import Board
from alphaomok.engine.gamemanager import GameManager
from alphaomok.models.policy_value_net import PolicyValueNet
from alphaomok.players.player import Player
from alphaomok.players.smart_player import SmartPlayer
from alphaomok.players.simpleplayer import SimplePlayer
from alphaomok.players.randomplayer import RandomPlayer
from alphaomok.players.humanplayer import HumanPlayer


@dataclass
class AnalysisResult:
    """Dataclass holding neural evaluation and candidate move analysis."""
    win_probability: float
    best_move: Tuple[int, int]
    top_moves: List[Tuple[Tuple[int, int], float]]

    def summary(self) -> str:
        """Returns formatted string representation of analysis results."""
        lines = [
            f"Win Probability: {self.win_probability * 100:.2f}%",
            f"Best Move: ({self.best_move[0]}, {self.best_move[1]})",
            "Top Candidate Moves:"
        ]
        for rank, ((r, c), p) in enumerate(self.top_moves, 1):
            lines.append(f"  {rank}. ({r:2d}, {c:2d}) - {p * 100:5.2f}%")
        return "\n".join(lines)


class AlphaOmok:
    """
    High-level AlphaOmok AI Player & Board Position Analyzer.
    Loads trained reinforced ResNet Policy-Value model and PUCT MCTS engine.
    """

    def __init__(self,
                 player_id: int = 0,
                 weights_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initializes AlphaOmok AI engine.
        Automatically resolves bundled weights from alphaomok/models/weights/rl_model_best.pt
        if weights_path is not specified.
        """
        self.player_id = player_id

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Resolve weights path
        if weights_path is None:
            pkg_dir = os.path.dirname(os.path.abspath(__file__))
            bundled = os.path.join(pkg_dir, "models", "weights", "rl_model_best.pt")
            if os.path.exists(bundled):
                weights_path = bundled
            else:
                alt = os.path.join(pkg_dir, "..", "models", "checkpoints", "rl_model_best.pt")
                if os.path.exists(alt):
                    weights_path = os.path.abspath(alt)
                else:
                    weights_path = bundled

        self.weights_path = weights_path

        # Initialize SmartPlayer engine with weights
        self.player = SmartPlayer(
            player_id=self.player_id,
            weights_path=self.weights_path,
            device=str(self.device)
        )
        self.model = self.player.model

    def get_move(self, board: Union[List[List[int]], Board], time_left_ms: float = 60000.0) -> Tuple[int, int]:
        """
        Computes best legal move given current board grid and remaining time.
        board: 19x19 nested list (-1 empty, 0 black, 1 white) or Board instance.
        time_left_ms: remaining clock in milliseconds.
        Returns: (row, col)
        """
        if isinstance(board, Board):
            grid = board.get_grid()
        else:
            grid = board
        return self.player.take_turn(grid, time_left_ms)

    def analyze(self, board: Union[List[List[int]], Board]) -> AnalysisResult:
        """
        Performs neural policy and value evaluation on the given board position.
        Computes policy distribution over legal moves, extracts top-5 moves with probabilities,
        and converts value [-1.0, +1.0] to win probability [0.0, 1.0].
        """
        if isinstance(board, Board):
            board_obj = board
        else:
            board_obj = Board(board)

        # Model inference
        probs, val = self.model.predict(board_obj, self.player_id, self.device)

        # Value to win probability
        win_prob = float(np.clip((val + 1.0) / 2.0, 0.0, 1.0))

        # Extract legal moves and sort by prior probability
        legal_moves_with_probs = []
        for r in range(19):
            for c in range(19):
                if board_obj.is_empty(r, c):
                    p = float(probs[r * 19 + c])
                    legal_moves_with_probs.append(((r, c), p))

        legal_moves_with_probs.sort(key=lambda x: x[1], reverse=True)
        top_5 = legal_moves_with_probs[:5]
        best_move = top_5[0][0] if top_5 else (-1, -1)

        return AnalysisResult(
            win_probability=win_prob,
            best_move=best_move,
            top_moves=top_5
        )


class Game:
    """
    High-level interactive Omok++ Game Session Manager.
    Supports turn-by-turn play, AI stepping, rule checking, and timer management.
    """

    def __init__(self,
                 black_player: Union[str, Player] = "smart",
                 white_player: Union[str, Player] = "human",
                 time_limit_minutes: int = 1):
        """
        Initializes new game session.
        black_player: 'smart', 'simple', 'random', 'human', or Player instance.
        white_player: 'smart', 'simple', 'random', 'human', or Player instance.
        time_limit_minutes: Clock budget in minutes (-1 or 0 for unlimited).
        """
        self.time_limit_minutes = time_limit_minutes
        self.time_limit_ms = float(time_limit_minutes * 60 * 1000) if time_limit_minutes > 0 else -1.0
        self.remaining_time = {
            Board.BLACK: self.time_limit_ms,
            Board.WHITE: self.time_limit_ms
        }

        self.players = {
            Board.BLACK: self._resolve_player(black_player, Board.BLACK),
            Board.WHITE: self._resolve_player(white_player, Board.WHITE)
        }

        self.board = Board()
        self.current_player = Board.BLACK
        self.game_over = False
        self.winner: Optional[int] = None
        self.reason: Optional[str] = None
        self.history: List[Tuple[int, int, int]] = []

    def _resolve_player(self, player_spec: Union[str, Player], player_id: int) -> Player:
        if isinstance(player_spec, Player):
            return player_spec
        if isinstance(player_spec, str):
            p = player_spec.strip().lower()
            if p == "smart":
                return SmartPlayer(player_id)
            elif p == "simple":
                return SimplePlayer(player_id)
            elif p == "random":
                return RandomPlayer(player_id)
            elif p == "human":
                return HumanPlayer(player_id)
            else:
                raise ValueError(f"Unknown player type '{player_spec}'. Choose from: 'smart', 'simple', 'random', 'human'.")
        raise TypeError(f"Invalid player type: {type(player_spec)}")

    def play_move(self, row: int, col: int) -> bool:
        """
        Plays a move at (row, col) for the current player.
        Returns True if the move was legal and applied, False otherwise.
        """
        if self.game_over:
            return False

        if not (0 <= row < 19 and 0 <= col < 19):
            return False

        if not self.board.is_empty(row, col):
            return False

        # Apply move
        self.board.place_stone(row, col, self.current_player)
        self.history.append((self.current_player, row, col))

        # Check win/draw
        is_over, winner = self.board.check_game_status()
        if is_over:
            self.game_over = True
            self.winner = winner
        else:
            self.current_player = 1 - self.current_player

        return True

    def step_ai(self) -> Tuple[Tuple[int, int], bool, Optional[int]]:
        """
        Asks current player to calculate and execute move.
        Returns:
            ((row, col), is_game_over, winner)
        """
        if self.game_over:
            return ((-1, -1), True, self.winner)

        current = self.current_player
        player_obj = self.players[current]
        time_left = self.remaining_time[current]

        start_t = time.perf_counter()
        move = player_obj.take_turn(self.board.get_grid(), time_left)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        if self.time_limit_ms > 0:
            self.remaining_time[current] -= elapsed_ms
            if self.remaining_time[current] <= 0:
                self.game_over = True
                self.winner = 1 - current
                self.reason = f"Player {current} timed out."
                return (move, True, self.winner)

        if not (isinstance(move, (tuple, list)) and len(move) == 2):
            self.game_over = True
            self.winner = 1 - current
            self.reason = f"Player {current} made invalid move format: {move}"
            return ((-1, -1), True, self.winner)

        r, c = move
        if not (0 <= r < 19 and 0 <= c < 19) or not self.board.is_empty(r, c):
            self.game_over = True
            self.winner = 1 - current
            self.reason = f"Player {current} made illegal move ({r}, {c})"
            return (move, True, self.winner)

        # Apply valid move
        self.board.place_stone(r, c, current)
        self.history.append((current, r, c))

        is_over, winner = self.board.check_game_status()
        if is_over:
            self.game_over = True
            self.winner = winner
        else:
            self.current_player = 1 - self.current_player

        return (move, self.game_over, self.winner)

    def get_board(self) -> List[List[int]]:
        """Returns 19x19 grid snapshot."""
        return self.board.get_grid()

    def get_winner(self) -> Optional[int]:
        """Returns winner (0: Black, 1: White, -1: Tie, None: Ongoing)."""
        return self.winner

    def is_game_over(self) -> bool:
        """Returns whether the game is over."""
        return self.game_over

    def print_board(self):
        """Displays ASCII board representation."""
        print(self.board.display())

    def play_game(self, verbose: bool = True) -> Tuple[Optional[int], str]:
        """Runs the entire game using GameManager and returns (winner, reason)."""
        mgr = GameManager(
            self.players[Board.BLACK],
            self.players[Board.WHITE],
            self.time_limit_minutes,
            verbose=verbose
        )
        winner, reason = mgr.play_game()
        self.board = mgr.board
        self.game_over = True
        self.winner = winner
        self.reason = reason
        return winner, reason
