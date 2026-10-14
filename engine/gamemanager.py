"""
GameManager for Omok++.
Manages game loop, timing, turn alternation, and outcome declarations.
"""

import time
from typing import Optional, Tuple
from engine.board import Board
from players.player import Player


class GameManager:
    def __init__(self, player0: Player, player1: Player, time_limit_minutes: int, verbose: bool = True):
        """
        player0: Player with Black stones (id=0)
        player1: Player with White stones (id=1)
        time_limit_minutes: total time in minutes per player (-1 for unlimited)
        """
        self.players = {Board.BLACK: player0, Board.WHITE: player1}
        self.board = Board()
        self.verbose = verbose

        if time_limit_minutes < 0:
            self.time_limit_ms = -1.0
            self.remaining_time = {Board.BLACK: -1.0, Board.WHITE: -1.0}
        else:
            self.time_limit_ms = float(time_limit_minutes * 60 * 1000)
            self.remaining_time = {
                Board.BLACK: self.time_limit_ms,
                Board.WHITE: self.time_limit_ms
            }

    def play_game(self) -> Tuple[Optional[int], str]:
        """
        Runs the game to completion.
        Returns:
            (winner: Optional[int], reason: str)
            winner: 0 (Black), 1 (White), -1 (Tie)
        """
        current_player = Board.BLACK
        turn_count = 0

        if self.verbose:
            print("=== Omok++ Game Started ===")
            print(f"Player 0 (Black): {self.players[Board.BLACK].__class__.__name__}")
            print(f"Player 1 (White): {self.players[Board.WHITE].__class__.__name__}")
            if self.time_limit_ms > 0:
                print(f"Time limit: {self.time_limit_ms / 60000:.1f} min ({self.time_limit_ms:.0f} ms) per player")
            else:
                print("Time limit: Unlimited")
            print(self.board.display())
            print()

        while True:
            turn_count += 1
            player_obj = self.players[current_player]
            color_name = "Black" if current_player == Board.BLACK else "White"
            time_left = self.remaining_time[current_player]

            # Get player move with timing
            board_snapshot = self.board.get_grid()

            start_t = time.perf_counter()
            try:
                move = player_obj.take_turn(board_snapshot, time_left)
            except Exception as e:
                winner = 1 - current_player
                reason = f"{color_name} crashed with exception: {e}"
                if self.verbose:
                    print(f"\n[ERROR] {reason}")
                    print(f"Winner: {'White' if winner == 1 else 'Black'}")
                return winner, reason

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            # Deduct time if time limit is active
            if self.time_limit_ms > 0:
                self.remaining_time[current_player] -= elapsed_ms
                if self.remaining_time[current_player] <= 0:
                    winner = 1 - current_player
                    reason = f"{color_name} ran out of time! (Elapsed: {elapsed_ms:.1f}ms, Left: {self.remaining_time[current_player]:.1f}ms)"
                    if self.verbose:
                        print(f"\n[TIMEOUT] {reason}")
                        print(f"Winner: {'White' if winner == 1 else 'Black'}")
                    return winner, reason

            # Validate move
            if not isinstance(move, tuple) and not isinstance(move, list) or len(move) != 2:
                winner = 1 - current_player
                reason = f"{color_name} returned invalid move format: {move}"
                if self.verbose:
                    print(f"\n[ILLEGAL MOVE] {reason}")
                return winner, reason

            r, c = move
            if not self.board.is_empty(r, c):
                winner = 1 - current_player
                reason = f"{color_name} played illegal move ({r}, {c}) - already occupied or out of bounds."
                if self.verbose:
                    print(f"\n[ILLEGAL MOVE] {reason}")
                return winner, reason

            # Apply move
            captured = self.board.place_stone(r, c, current_player)

            if self.verbose:
                cap_str = f" [Captured {len(captured)} stones!]" if captured else ""
                time_str = f" (Time: {elapsed_ms:.1f}ms, Remaining: {self.remaining_time[current_player]/1000:.1f}s)" if self.time_limit_ms > 0 else f" ({elapsed_ms:.1f}ms)"
                print(f"Turn {turn_count}: {color_name} plays ({r}, {c}){cap_str}{time_str}")
                print(self.board.display())
                print()

            # Check game status
            is_over, winner = self.board.check_game_status()
            if is_over:
                if winner == -1:
                    reason = "Game ended in a tie!"
                    if self.verbose:
                        print(f"\n=== Result: TIE! ===")
                elif winner == Board.BLACK:
                    reason = "Black wins with exact 5-in-a-row!"
                    if self.verbose:
                        print(f"\n=== Winner: Black (Player 0)! ===")
                elif winner == Board.WHITE:
                    reason = "White wins with exact 5-in-a-row!"
                    if self.verbose:
                        print(f"\n=== Winner: White (Player 1)! ===")
                return winner, reason

            # Switch turns
            current_player = 1 - current_player
