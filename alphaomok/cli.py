"""
AlphaOmok Command Line Interface (CLI).
Commands:
  alphaomok play: Launches interactive GUI
  alphaomok match [--black ...] [--white ...] [--time ...]: Runs console match
  alphaomok analyze: Prints neural evaluation of board position
  alphaomok info: Prints SDK architecture and training info
"""

import argparse
import sys
from typing import Optional, List


def cmd_play(args):
    """Launches interactive desktop GUI."""
    print("Launching AlphaOmok Interactive GUI Arena...")
    try:
        from alphaomok import launch_gui
        launch_gui()
    except Exception as e:
        print(f"[Error] Failed to launch GUI: {e}")
        print("Note: In headless or non-display environments, GUI cannot be launched.")
        print("Use 'alphaomok match' to play or simulate matches in terminal.")
        return 1
    return 0


def cmd_match(args):
    """Runs a console match between two players."""
    from alphaomok.api import Game

    print("==================================================================")
    print(f"  AlphaOmok Console Match: {args.black.upper()} vs {args.white.upper()}")
    print(f"  Time Control: {args.time} minute(s) per player")
    print("==================================================================")

    game = Game(
        black_player=args.black,
        white_player=args.white,
        time_limit_minutes=args.time
    )
    winner, reason = game.play_game(verbose=True)
    print("==================================================================")
    print(f"Final Result: {reason}")
    if winner == 0:
        print("Winner: BLACK (Player 0)")
    elif winner == 1:
        print("Winner: WHITE (Player 1)")
    else:
        print("Match ended in a DRAW")
    print("==================================================================")
    return 0


def cmd_analyze(args):
    """Prints neural evaluation of starting/sample board position."""
    from alphaomok.api import AlphaOmok
    from alphaomok.engine.board import Board

    print("==================================================================")
    print("               AlphaOmok Neural Position Analyzer                 ")
    print("==================================================================")
    print("Evaluating 19x19 Omok++ Opening Position (Player: Black, Turn 1)...")

    board = Board()
    ai = AlphaOmok(player_id=Board.BLACK)
    result = ai.analyze(board)

    print("\n" + result.summary())
    print("==================================================================")
    return 0


def cmd_info(args):
    """Prints SDK version, ResNet architecture, and training statistics."""
    from alphaomok import __version__

    info_text = f"""
================================================================================
                           AlphaOmok SDK v{__version__}
               Reinforced Deep ResNet & PUCT MCTS Engine for Omok++
================================================================================
SDK Version:           {__version__}
Rule Engine:           Omok++ (Official 19x19 Grid)
                       - Custodial Capture: Sandwich two opponent stones (X-O-O-X)
                       - Exact 5-in-a-Row Win: Exactly 5 stones required (6+ is overline)
                       - Time Control: 60-second tournament budget constraint

Neural Architecture:   Deep Residual Network (Policy-Value Dual Head)
                       - Residual Blocks: 10 ResBlocks
                       - Feature Channels: 128 Channels
                       - Total Parameters: 7,879,018 (~7.88M parameters)
                       - Input Representation: 7-Channel 19x19 Spatial Tensor

Training Statistics:   Multi-Node Distributed Reinforcement Learning
                       - Reinforcement Iterations: 23 Iterations
                       - Training Dataset: 2,230,000 (~2.23M positions evaluated)
                       - Self-Play Engine: Numba JIT C-speed Native Simulation

Hardware Optimization: CUDA, MPS, CPU auto-dispatching
                       Zero-copy Numba JIT bare-metal state representation
================================================================================
"""
    print(info_text.strip())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alphaomok",
        description="AlphaOmok: Reinforced Neural MCTS Engine & SDK for Omok++"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # play
    p_play = subparsers.add_parser("play", help="Launch interactive Omok++ GUI arena")
    p_play.set_defaults(func=cmd_play)

    # match
    p_match = subparsers.add_parser("match", help="Run a console Omok++ match")
    p_match.add_argument("--black", choices=["smart", "simple", "random", "human"],
                         default="smart", help="Player controlling Black stones (default: smart)")
    p_match.add_argument("--white", choices=["smart", "simple", "random", "human"],
                         default="simple", help="Player controlling White stones (default: simple)")
    p_match.add_argument("--time", type=int, default=1,
                         help="Time limit per player in minutes (default: 1)")
    p_match.set_defaults(func=cmd_match)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Evaluate starting/sample board position")
    p_analyze.set_defaults(func=cmd_analyze)

    # info
    p_info = subparsers.add_parser("info", help="Display SDK architecture and training info")
    p_info.set_defaults(func=cmd_info)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
