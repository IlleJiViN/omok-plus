"""
Comprehensive Test Suite for the AlphaOmok Python SDK.
Verifies:
1. import alphaomok and version string.
2. ai = alphaomok.AlphaOmok(0) auto-loading embedded weights.
3. ai.get_move(grid, 60000.0) returning valid legal move.
4. ai.analyze(grid) returning valid AnalysisResult (win_probability in [0.0, 1.0], top 5 moves).
5. game = alphaomok.Game() initialization, play_move, and step_ai.
6. CLI help and subcommands invocation.
"""

import os
import sys
import subprocess
import unittest

# Ensure project directory is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestAlphaOmokSDK(unittest.TestCase):

    def test_01_import_and_version(self):
        """Verify package import, version, and exposed public symbols."""
        import alphaomok

        self.assertEqual(alphaomok.__version__, "1.0.0")

        # Required public symbols
        self.assertTrue(hasattr(alphaomok, "AlphaOmok"))
        self.assertTrue(hasattr(alphaomok, "Game"))
        self.assertTrue(hasattr(alphaomok, "Board"))
        self.assertTrue(hasattr(alphaomok, "AnalysisResult"))
        self.assertTrue(hasattr(alphaomok, "SmartPlayer"))
        self.assertTrue(hasattr(alphaomok, "SimplePlayer"))
        self.assertTrue(hasattr(alphaomok, "RandomPlayer"))
        self.assertTrue(hasattr(alphaomok, "HumanPlayer"))
        self.assertTrue(hasattr(alphaomok, "analyze"))
        self.assertTrue(hasattr(alphaomok, "launch_gui"))
        print("[PASS] Test 1: Package imports and public symbols verified.")

    def test_02_alphaomok_initialization_and_weights(self):
        """Verify AlphaOmok AI initialization and automatic embedded weights resolution."""
        import alphaomok

        ai = alphaomok.AlphaOmok(player_id=0)
        self.assertEqual(ai.player_id, 0)
        self.assertIsNotNone(ai.weights_path)
        self.assertTrue(os.path.exists(ai.weights_path), f"Weights path does not exist: {ai.weights_path}")
        self.assertTrue(ai.weights_path.endswith("rl_model_best.pt"))

        # Verify model structure and parameter count (~7.88M)
        self.assertIsNotNone(ai.model)
        total_params = sum(p.numel() for p in ai.model.parameters())
        self.assertEqual(total_params, 7879018, f"Unexpected parameter count: {total_params}")
        print(f"[PASS] Test 2: AlphaOmok initialized with bundled weights ({total_params:,} parameters).")

    def test_03_get_move(self):
        """Verify ai.get_move(grid, time_left) returns a valid legal coordinate."""
        import alphaomok

        ai = alphaomok.AlphaOmok(player_id=0)
        grid = [[-1 for _ in range(19)] for _ in range(19)]

        # Move 1: Empty board
        move = ai.get_move(grid, time_left_ms=60000.0)
        self.assertIsInstance(move, tuple)
        self.assertEqual(len(move), 2)
        r, c = move
        self.assertTrue(0 <= r < 19 and 0 <= c < 19, f"Move out of bounds: ({r}, {c})")
        self.assertEqual(grid[r][c], -1, "Move played on non-empty cell")

        # Move 2: Board with some stones placed
        grid[9][9] = 0  # Black
        grid[9][10] = 1 # White
        move2 = ai.get_move(grid, time_left_ms=58000.0)
        self.assertIsInstance(move2, tuple)
        r2, c2 = move2
        self.assertTrue(0 <= r2 < 19 and 0 <= c2 < 19)
        self.assertEqual(grid[r2][c2], -1)
        self.assertNotEqual((r2, c2), (9, 9))
        self.assertNotEqual((r2, c2), (9, 10))
        print(f"[PASS] Test 3: Legal moves returned successfully: {move}, {move2}.")

    def test_04_analyze(self):
        """Verify ai.analyze(grid) returns valid AnalysisResult with probability and top moves."""
        import alphaomok

        ai = alphaomok.AlphaOmok(player_id=0)
        grid = [[-1 for _ in range(19)] for _ in range(19)]

        result = ai.analyze(grid)
        self.assertIsInstance(result, alphaomok.AnalysisResult)

        # Win probability between 0.0 and 1.0
        self.assertGreaterEqual(result.win_probability, 0.0)
        self.assertLessEqual(result.win_probability, 1.0)

        # Best move valid
        self.assertIsInstance(result.best_move, tuple)
        self.assertEqual(len(result.best_move), 2)
        br, bc = result.best_move
        self.assertTrue(0 <= br < 19 and 0 <= bc < 19)

        # Top 5 moves
        self.assertEqual(len(result.top_moves), 5)
        for (r, c), prob in result.top_moves:
            self.assertTrue(0 <= r < 19 and 0 <= c < 19)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

        # Best move matches first top move
        self.assertEqual(result.best_move, result.top_moves[0][0])

        # Summary string
        summary = result.summary()
        self.assertIsInstance(summary, str)
        self.assertIn("Win Probability", summary)
        self.assertIn("Best Move", summary)
        self.assertIn("Top Candidate Moves", summary)

        # Also test module-level analyze() function
        mod_result = alphaomok.analyze(grid, player_id=1)
        self.assertIsInstance(mod_result, alphaomok.AnalysisResult)
        self.assertGreaterEqual(mod_result.win_probability, 0.0)
        self.assertLessEqual(mod_result.win_probability, 1.0)
        print(f"[PASS] Test 4: Position analysis verified (WinProb: {result.win_probability*100:.2f}%, Best: {result.best_move}).")

    def test_05_game_session(self):
        """Verify Game session initialization, play_move, and step_ai execution."""
        import alphaomok

        # Black: Smart, White: Simple
        game = alphaomok.Game(black_player="smart", white_player="simple", time_limit_minutes=1)
        self.assertFalse(game.is_game_over())
        self.assertIsNone(game.get_winner())

        # Step 1: AI (Black) takes first move
        move1, is_over, winner = game.step_ai()
        self.assertFalse(is_over)
        self.assertIsNone(winner)
        r1, c1 = move1
        self.assertTrue(0 <= r1 < 19 and 0 <= c1 < 19)
        board = game.get_board()
        self.assertEqual(board[r1][c1], 0) # Black placed

        # Step 2: Next turn is White (Simple)
        move2, is_over2, winner2 = game.step_ai()
        self.assertFalse(is_over2)
        self.assertIsNone(winner2)
        r2, c2 = move2
        self.assertTrue(0 <= r2 < 19 and 0 <= c2 < 19)
        board2 = game.get_board()
        self.assertEqual(board2[r2][c2], 1) # White placed

        # Test play_move on a new Human vs AI game
        game_interactive = alphaomok.Game(black_player="human", white_player="smart", time_limit_minutes=1)
        # Black (human) plays (9, 9)
        success = game_interactive.play_move(9, 9)
        self.assertTrue(success)
        self.assertEqual(game_interactive.get_board()[9][9], 0)

        # Illegal move to occupied cell returns False
        fail = game_interactive.play_move(9, 9)
        self.assertFalse(fail)

        # Out-of-bounds move returns False
        fail_oob = game_interactive.play_move(20, 20)
        self.assertFalse(fail_oob)

        # Now step AI (White)
        ai_move, over, w = game_interactive.step_ai()
        self.assertTrue(0 <= ai_move[0] < 19 and 0 <= ai_move[1] < 19)
        self.assertEqual(game_interactive.get_board()[ai_move[0]][ai_move[1]], 1)

        print("[PASS] Test 5: Game session management and step_ai verified.")

    def test_06_cli_invocation(self):
        """Verify CLI entry points (--help, info, analyze)."""
        # Test 6a: python -m alphaomok.cli --help
        res_help = subprocess.run(
            [sys.executable, "-m", "alphaomok.cli", "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res_help.returncode, 0)
        self.assertIn("alphaomok", res_help.stdout)
        self.assertIn("play", res_help.stdout)
        self.assertIn("match", res_help.stdout)
        self.assertIn("analyze", res_help.stdout)
        self.assertIn("info", res_help.stdout)

        # Test 6b: python -m alphaomok.cli info
        res_info = subprocess.run(
            [sys.executable, "-m", "alphaomok.cli", "info"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res_info.returncode, 0)
        self.assertIn("10 ResBlocks", res_info.stdout)
        self.assertIn("128 Channels", res_info.stdout)
        self.assertIn("7,879,018", res_info.stdout)
        self.assertIn("23 Iterations", res_info.stdout)

        # Test 6c: python -m alphaomok.cli analyze
        res_ana = subprocess.run(
            [sys.executable, "-m", "alphaomok.cli", "analyze"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res_ana.returncode, 0)
        self.assertIn("Win Probability", res_ana.stdout)
        self.assertIn("Best Move", res_ana.stdout)

        print("[PASS] Test 6: CLI subcommands (--help, info, analyze) verified.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
