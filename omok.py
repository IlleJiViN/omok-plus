"""
Main entry point for Omok++ game.
Usage:
    python omok.py [player-1] [player-2] [time-limit]

Player codes:
    0: HumanPlayer
    1: RandomPlayer
    2: SimplePlayer
    3: SmartPlayer

time-limit:
    Integer number of minutes for the player throughout the entire game (-1 for unlimited).
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.gamemanager import GameManager
from players.humanplayer import HumanPlayer
from players.randomplayer import RandomPlayer
from players.simpleplayer import SimplePlayer

try:
    from players.smart_player import SmartPlayer
except ImportError:
    # Fallback to SimplePlayer if SmartPlayer is not yet initialized
    SmartPlayer = SimplePlayer


def get_player(player_code: int, player_id: int):
    if player_code == 0:
        return HumanPlayer(player_id)
    elif player_code == 1:
        return RandomPlayer(player_id)
    elif player_code == 2:
        return SimplePlayer(player_id)
    elif player_code == 3:
        return SmartPlayer(player_id)
    else:
        raise ValueError(f"Unknown player code: {player_code}. Choose 0, 1, 2, or 3.")


def main():
    if len(sys.argv) < 4:
        print("Usage: python omok.py [player-1] [player-2] [time-limit]")
        print("  player-1, player-2: 0 (Human), 1 (Random), 2 (Simple), 3 (Smart)")
        print("  time-limit: minutes per player (-1 for unlimited)")
        sys.exit(1)

    try:
        p1_code = int(sys.argv[1])
        p2_code = int(sys.argv[2])
        time_limit = int(sys.argv[3])
    except ValueError as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)

    player0 = get_player(p1_code, 0)
    player1 = get_player(p2_code, 1)

    manager = GameManager(player0, player1, time_limit, verbose=True)
    winner, reason = manager.play_game()
    print(f"Game Over: {reason}")


if __name__ == "__main__":
    main()
