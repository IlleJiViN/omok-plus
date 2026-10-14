"""
Unit tests for Omok++ rules and game engine.
Validates all cases in Figure 1 (Winning vs Non-winning) and Figure 2 (Captures).
"""

import unittest
from engine.board import Board


class TestOmokRules(unittest.TestCase):

    def test_horizontal_win(self):
        # Figure 1a: Exactly 5 horizontal stones
        b = Board()
        for c in range(4):
            b.place_stone(5, c, Board.BLACK)
        captured = b.place_stone(5, 4, Board.BLACK)
        self.assertEqual(captured, [])
        is_over, winner = b.check_game_status()
        self.assertTrue(is_over)
        self.assertEqual(winner, Board.BLACK)

    def test_vertical_win(self):
        # Figure 1b: Exactly 5 vertical stones
        b = Board()
        for r in range(4):
            b.place_stone(r, 7, Board.WHITE)
        b.place_stone(4, 7, Board.WHITE)
        is_over, winner = b.check_game_status()
        self.assertTrue(is_over)
        self.assertEqual(winner, Board.WHITE)

    def test_diagonal_win(self):
        # Figure 1c: Exactly 5 diagonal stones
        b = Board()
        for i in range(5):
            b.place_stone(2 + i, 3 + i, Board.BLACK)
        is_over, winner = b.check_game_status()
        self.assertTrue(is_over)
        self.assertEqual(winner, Board.BLACK)

    def test_two_fives_simultaneous_same_player(self):
        # Figure 1d: Placing one stone forms two 5-in-a-rows for the same player
        b = Board()
        # horizontal: (5, 0..3)
        for c in range(4):
            b.place_stone(5, c, Board.BLACK)
        # vertical: (1..4, 4)
        for r in range(1, 5):
            b.place_stone(r, 4, Board.BLACK)
        # place at intersection (5, 4)
        b.place_stone(5, 4, Board.BLACK)
        self.assertEqual(b.count_exact_fives(Board.BLACK), 2)
        is_over, winner = b.check_game_status()
        self.assertTrue(is_over)
        self.assertEqual(winner, Board.BLACK)

    def test_six_in_a_row_not_win(self):
        # Figure 1f: 6 contiguous stones does NOT win
        b = Board()
        for c in range(6):
            b.place_stone(5, c, Board.BLACK)
        self.assertEqual(b.count_exact_fives(Board.BLACK), 0)
        is_over, winner = b.check_game_status()
        self.assertFalse(is_over)
        self.assertIsNone(winner)

    def test_seven_in_a_row_not_win(self):
        b = Board()
        for c in range(7):
            b.place_stone(8, c, Board.WHITE)
        self.assertEqual(b.count_exact_fives(Board.WHITE), 0)
        is_over, winner = b.check_game_status()
        self.assertFalse(is_over)

    def test_horizontal_capture(self):
        # Figure 2a: Horizontal capture
        # Initial: B(5, 0), W(5, 1), W(5, 2), empty(5, 3)
        # Move: B plays (5, 3) -> captures W(5, 1) and W(5, 2)
        b = Board()
        b.place_stone(5, 0, Board.BLACK)
        b.place_stone(5, 1, Board.WHITE)
        b.place_stone(5, 2, Board.WHITE)
        captured = b.place_stone(5, 3, Board.BLACK)
        self.assertIn((5, 1), captured)
        self.assertIn((5, 2), captured)
        self.assertEqual(len(captured), 2)
        self.assertEqual(b.grid[5][1], Board.EMPTY)
        self.assertEqual(b.grid[5][2], Board.EMPTY)
        self.assertEqual(b.captures[Board.BLACK], 2)

    def test_vertical_capture(self):
        # Figure 2b: Vertical capture
        b = Board()
        b.place_stone(0, 5, Board.WHITE)
        b.place_stone(1, 5, Board.BLACK)
        b.place_stone(2, 5, Board.BLACK)
        captured = b.place_stone(3, 5, Board.WHITE)
        self.assertEqual(len(captured), 2)
        self.assertEqual(b.grid[1][5], Board.EMPTY)
        self.assertEqual(b.grid[2][5], Board.EMPTY)
        self.assertEqual(b.captures[Board.WHITE], 2)

    def test_diagonal_capture(self):
        # Figure 2c: Diagonal capture
        b = Board()
        b.place_stone(1, 1, Board.BLACK)
        b.place_stone(2, 2, Board.WHITE)
        b.place_stone(3, 3, Board.WHITE)
        captured = b.place_stone(4, 4, Board.BLACK)
        self.assertEqual(len(captured), 2)
        self.assertEqual(b.grid[2][2], Board.EMPTY)
        self.assertEqual(b.grid[3][3], Board.EMPTY)

    def test_multiple_captures_simultaneous(self):
        # Figure 2d: Placing one stone captures multiple pairs simultaneously
        b = Board()
        # Horizontal pair to left: B(5, 0), W(5, 1), W(5, 2), [B at (5, 3)]
        b.place_stone(5, 0, Board.BLACK)
        b.place_stone(5, 1, Board.WHITE)
        b.place_stone(5, 2, Board.WHITE)
        # Vertical pair above: B(0, 3), W(1, 3), W(2, 3), [B at (5, 3)]? Wait, (2, 3) to (5, 3) is 3 steps: 0, 1, 2, 3.
        # B at (0, 3) is r-3. Wait, if placed at (3, 3), then r-1=(2,3), r-2=(1,3), r-3=(0,3).
        # Let's align center at (3, 3):
        # Horizontal left: B(3, 0), W(3, 1), W(3, 2)
        # Vertical above: B(0, 3), W(1, 3), W(2, 3)
        # Horizontal right: W(3, 4), W(3, 5), B(3, 6)
        # Vertical below: W(4, 3), W(5, 3), B(6, 3)
        b = Board()
        b.place_stone(3, 0, Board.BLACK)
        b.place_stone(3, 1, Board.WHITE)
        b.place_stone(3, 2, Board.WHITE)

        b.place_stone(0, 3, Board.BLACK)
        b.place_stone(1, 3, Board.WHITE)
        b.place_stone(2, 3, Board.WHITE)

        b.place_stone(3, 6, Board.BLACK)
        b.place_stone(3, 5, Board.WHITE)
        b.place_stone(3, 4, Board.WHITE)

        b.place_stone(6, 3, Board.BLACK)
        b.place_stone(5, 3, Board.WHITE)
        b.place_stone(4, 3, Board.WHITE)

        captured = b.place_stone(3, 3, Board.BLACK)
        self.assertEqual(len(captured), 8)
        self.assertEqual(b.captures[Board.BLACK], 8)
        # All 4 pairs removed
        for r, c in [(3,1), (3,2), (1,3), (2,3), (3,4), (3,5), (4,3), (5,3)]:
            self.assertEqual(b.grid[r][c], Board.EMPTY)

    def test_middle_placement_not_capture(self):
        # Figure 2e: Placing a stone in the middle between opponent stones does NOT take!
        # Configuration: W(5, 0), empty(5, 1), W(5, 2), W(5, 3)
        # B plays at (5, 1). B is placed between W(5, 0) and W(5, 2).
        # Neither White nor Black takes!
        b = Board()
        b.place_stone(5, 0, Board.WHITE)
        b.place_stone(5, 2, Board.WHITE)
        b.place_stone(5, 3, Board.WHITE)
        captured = b.place_stone(5, 1, Board.BLACK)
        self.assertEqual(len(captured), 0)
        self.assertEqual(b.grid[5][0], Board.WHITE)
        self.assertEqual(b.grid[5][2], Board.WHITE)
        self.assertEqual(b.grid[5][3], Board.WHITE)

    def test_three_stones_not_captured(self):
        # 3 stones cannot be captured (only exactly 2 contiguous stones)
        b = Board()
        b.place_stone(5, 0, Board.BLACK)
        b.place_stone(5, 1, Board.WHITE)
        b.place_stone(5, 2, Board.WHITE)
        b.place_stone(5, 3, Board.WHITE)
        captured = b.place_stone(5, 4, Board.BLACK)
        self.assertEqual(len(captured), 0)

    def test_capture_turns_six_into_five_opponent_wins(self):
        # Spec rule: "if you take one of the stones forming a 6-in-a-row so that the
        # taking move makes it a 5-in-a-row. Opponent wins."
        # White has 6 stones: (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6)
        # Currently 6 stones -> White does NOT win.
        # Suppose Black has a stone at (5, 9).
        # Wait, if Black has a stone at (5, 8)?
        # Let's say White has (5, 1..6) and White also has (5, 7)? No, we want White stones at (5, 6), (5, 7)?
        # For Black to capture (5, 6), Black needs: B at (5, 8), captures W at (5, 7) and (5, 6) by placing at (5, 5)?
        # But (5, 5) is occupied!
        # Instead, capture vertically!
        # White has horizontal 6-in-a-row: (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7)
        # Consider column 7: White has (5, 7) and White has (6, 7).
        # Black has (4, 7).
        # Black plays (7, 7)!
        # Then on column 7: B at (4, 7), W at (5, 7), W at (6, 7), B placed at (7, 7).
        # This captures W at (5, 7) and (6, 7)!
        # After (5, 7) is removed from the horizontal line:
        # The remaining horizontal White stones are (5, 2), (5, 3), (5, 4), (5, 5), (5, 6) -> EXACTLY 5 STONES!
        # So White now has an exact 5-in-a-row, and Black has no 5-in-a-row.
        # Result: White WINS on Black's turn!
        b = Board()
        # Place horizontal White stones from (5, 2) to (5, 7) [6 stones]
        for c in range(2, 8):
            b.grid[5][c] = Board.WHITE
        # Also place White at (6, 7)
        b.grid[6][7] = Board.WHITE
        # Place Black at (4, 7)
        b.grid[4][7] = Board.BLACK

        # Verify White has 6-in-a-row initially -> not winning
        is_over, winner = b.check_game_status()
        self.assertFalse(is_over)

        # Black plays at (7, 7)
        captured = b.place_stone(7, 7, Board.BLACK)
        self.assertIn((5, 7), captured)
        self.assertIn((6, 7), captured)

        # Now White's horizontal row has length 5: (5, 2) to (5, 6)
        self.assertEqual(b.count_exact_fives(Board.WHITE), 1)
        is_over, winner = b.check_game_status()
        self.assertTrue(is_over)
        self.assertEqual(winner, Board.WHITE)

    def test_simultaneous_five_results_in_tie(self):
        # Spec rule: "If both players have 5-in-a-row simultaneously as a result after a move, it is a tie."
        b = Board()
        # Black has (0, 0..3)
        for c in range(4):
            b.grid[0][c] = Board.BLACK
        # White has 6 stones: (5, 2..7)
        for c in range(2, 8):
            b.grid[5][c] = Board.WHITE
        # White at (6, 7)
        b.grid[6][7] = Board.WHITE

        # Wait, if Black completes 5-in-a-row AND captures at the same time:
        # Let Black's 5th stone be the capture stone!
        # Black has (0, 7), (1, 7), (2, 7), (3, 7).
        # White has (5, 7), (6, 7).
        # Black plays at (4, 7).
        # Then on column 7: Black has (0, 7), (1, 7), (2, 7), (3, 7), (4, 7) -> 5-in-a-row!
        # And downwards from (4, 7): W(5, 7), W(6, 7), and Black has (7, 7)!
        # So Black captures W(5, 7) and W(6, 7)!
        # Removing W(5, 7) reduces White's horizontal 6-in-a-row to 5-in-a-row: W(5, 2..6)!
        # Now BOTH Black and White have exact 5-in-a-row!
        b = Board()
        # Black vertical (0, 7) to (3, 7)
        for r in range(4):
            b.grid[r][7] = Board.BLACK
        # Black anchor at (7, 7)
        b.grid[7][7] = Board.BLACK

        # White horizontal (5, 2) to (5, 7)
        for c in range(2, 8):
            b.grid[5][c] = Board.WHITE
        # White at (6, 7)
        b.grid[6][7] = Board.WHITE

        # Black plays at (4, 7):
        # 1. Completes vertical 5-in-a-row (0..4, 7)
        # 2. Captures downward: (5, 7) and (6, 7) with anchor (7, 7)
        # 3. Capture removes (5, 7), leaving White with (5, 2..6) -> 5-in-a-row!
        captured = b.place_stone(4, 7, Board.BLACK)
        self.assertIn((5, 7), captured)
        self.assertIn((6, 7), captured)
        self.assertEqual(b.count_exact_fives(Board.BLACK), 1)
        self.assertEqual(b.count_exact_fives(Board.WHITE), 1)

        is_over, winner = b.check_game_status()
        self.assertTrue(is_over)
        self.assertEqual(winner, -1)  # TIE!


if __name__ == '__main__':
    unittest.main()
