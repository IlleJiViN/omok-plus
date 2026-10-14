"""
AlphaOmok: Reinforced Neural MCTS Engine & Interactive SDK for Omok++.

Exposes:
- AlphaOmok: High-level AI player and board analyzer
- Game: Interactive Omok++ game session manager
- Board: 19x19 board representation with Omok++ custodial capture & exact 5-in-a-row win
- AnalysisResult: Analysis dataclass
- SmartPlayer, SimplePlayer, RandomPlayer, HumanPlayer: Standard player implementations
- analyze(board, player_id=0, device=None) -> AnalysisResult
- launch_gui(): Launches interactive Omok++ GUI
"""

from typing import List, Optional, Union

from alphaomok.api import AlphaOmok, Game, AnalysisResult
from alphaomok.engine.board import Board
from alphaomok.players.smart_player import SmartPlayer
from alphaomok.players.simpleplayer import SimplePlayer
from alphaomok.players.randomplayer import RandomPlayer
from alphaomok.players.humanplayer import HumanPlayer

__version__ = "1.0.0"


def analyze(board: Union[List[List[int]], Board],
            player_id: int = 0,
            device: Optional[str] = None) -> AnalysisResult:
    """
    Convenience function to analyze a board state using the bundled AlphaOmok model.

    Args:
        board: 19x19 nested list or Board instance.
        player_id: 0 for Black, 1 for White.
        device: 'cpu', 'cuda', or None for auto-detection.

    Returns:
        AnalysisResult containing win_probability, best_move, and top_moves.
    """
    ai = AlphaOmok(player_id=player_id, device=device)
    return ai.analyze(board)


def launch_gui():
    """Launches interactive Omok++ desktop GUI."""
    try:
        from alphaomok.gui.omok_gui import launch
        launch()
    except Exception as e:
        print(f"Error launching GUI: {e}")
        raise


__all__ = [
    "AlphaOmok",
    "Game",
    "Board",
    "AnalysisResult",
    "SmartPlayer",
    "SimplePlayer",
    "RandomPlayer",
    "HumanPlayer",
    "analyze",
    "launch_gui",
    "__version__",
]
