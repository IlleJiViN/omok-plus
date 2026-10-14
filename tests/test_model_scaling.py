"""
Unit and integration tests for PolicyValueNet model scaling.
Verifies:
1. Instantiation of baseline vs 2x scaled models.
2. Total parameter count comparison.
3. Single and batched forward passes on CPU and CUDA.
4. Correctness of predict(board, player, device) on both models.
5. Checkpoint serialization and from_config loading, including backwards compatibility.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.board import Board
from models.policy_value_net import PolicyValueNet


class TestModelScaling(unittest.TestCase):

    def setUp(self):
        self.device_cpu = torch.device("cpu")
        self.cuda_available = torch.cuda.is_available()
        self.device_cuda = torch.device("cuda") if self.cuda_available else None

    def test_instantiation_and_configs(self):
        """1. Verify instantiation of baseline vs 2x scaled model and config dictionaries."""
        baseline = PolicyValueNet.create_baseline()
        self.assertEqual(baseline.in_channels, 7)
        self.assertEqual(baseline.num_channels, 64)
        self.assertEqual(baseline.num_res_blocks, 5)
        self.assertEqual(baseline.policy_channels, 32)
        self.assertEqual(baseline.value_channels, 16)

        config_base = baseline.get_config()
        self.assertEqual(config_base, {
            "in_channels": 7,
            "num_channels": 64,
            "num_res_blocks": 5,
            "policy_channels": 32,
            "value_channels": 16,
        })

        model_2x = PolicyValueNet.create_2x()
        self.assertEqual(model_2x.in_channels, 7)
        self.assertEqual(model_2x.num_channels, 128)
        self.assertEqual(model_2x.num_res_blocks, 10)
        self.assertEqual(model_2x.policy_channels, 32)
        self.assertEqual(model_2x.value_channels, 16)

        config_2x = model_2x.get_config()
        self.assertEqual(config_2x, {
            "in_channels": 7,
            "num_channels": 128,
            "num_res_blocks": 10,
            "policy_channels": 32,
            "value_channels": 16,
        })

        # Test custom parameters
        custom = PolicyValueNet(in_channels=9, num_channels=96, num_res_blocks=7,
                                policy_channels=48, value_channels=24)
        self.assertEqual(custom.num_channels, 96)
        self.assertEqual(custom.num_res_blocks, 7)
        self.assertEqual(custom.policy_channels, 48)
        self.assertEqual(custom.value_channels, 24)

    def test_parameter_count_comparison(self):
        """2. Verify parameter counts between baseline and 2x scaled models."""
        baseline = PolicyValueNet.create_baseline()
        model_2x = PolicyValueNet.create_2x()

        p_base = baseline.num_parameters
        p_2x = model_2x.num_parameters

        print(f"\n[Parameter Counts] Baseline: {p_base:,} | 2x Scaled: {p_2x:,} (+{p_2x - p_base:,})")

        self.assertEqual(p_base, 5_287_466)
        self.assertEqual(p_2x, 7_879_018)
        self.assertGreater(p_2x, p_base)
        # Backbone capacity increase (10 blocks x 128 channels vs 5 blocks x 64 channels)
        backbone_base = sum(p.numel() for p in baseline.res_blocks.parameters())
        backbone_2x = sum(p.numel() for p in model_2x.res_blocks.parameters())
        print(f"[Backbone Capacity] Baseline: {backbone_base:,} | 2x Scaled: {backbone_2x:,} ({backbone_2x / backbone_base:.2f}x)")
        self.assertGreater(backbone_2x, backbone_base * 7)

    def test_forward_cpu(self):
        """3a. Single and batched forward passes on CPU."""
        for name, model in [("Baseline", PolicyValueNet.create_baseline()),
                            ("2x Scaled", PolicyValueNet.create_2x())]:
            model.eval()
            # Single item pass
            x_single = torch.randn(1, 7, 19, 19, dtype=torch.float32)
            with torch.no_grad():
                p_single, v_single = model(x_single)
            self.assertEqual(p_single.shape, (1, 361), f"{name} single policy shape mismatch")
            self.assertEqual(v_single.shape, (1, 1), f"{name} single value shape mismatch")
            self.assertTrue(torch.all(v_single >= -1.0) and torch.all(v_single <= 1.0))
            self.assertFalse(torch.isnan(p_single).any())
            self.assertFalse(torch.isnan(v_single).any())

            # Batched pass (Batch size 8)
            x_batch = torch.randn(8, 7, 19, 19, dtype=torch.float32)
            with torch.no_grad():
                p_batch, v_batch = model(x_batch)
            self.assertEqual(p_batch.shape, (8, 361), f"{name} batched policy shape mismatch")
            self.assertEqual(v_batch.shape, (8, 1), f"{name} batched value shape mismatch")
            self.assertTrue(torch.all(v_batch >= -1.0) and torch.all(v_batch <= 1.0))

    def test_forward_cuda(self):
        """3b. Single and batched forward passes on CUDA (if available)."""
        if not self.cuda_available:
            self.skipTest("CUDA not available on this machine")

        print(f"\n[CUDA Benchmark] Testing forward passes on {torch.cuda.get_device_name(0)}")
        for name, model in [("Baseline", PolicyValueNet.create_baseline()),
                            ("2x Scaled", PolicyValueNet.create_2x())]:
            model = model.to(self.device_cuda)
            model.eval()

            # Single item
            x_single = torch.randn(1, 7, 19, 19, device=self.device_cuda)
            with torch.no_grad():
                p_single, v_single = model(x_single)
            self.assertEqual(p_single.shape, (1, 361))
            self.assertEqual(v_single.shape, (1, 1))
            self.assertEqual(p_single.device.type, "cuda")

            # Batched (Batch size 64)
            x_batch = torch.randn(64, 7, 19, 19, device=self.device_cuda)
            with torch.no_grad():
                p_batch, v_batch = model(x_batch)
            self.assertEqual(p_batch.shape, (64, 361))
            self.assertEqual(v_batch.shape, (64, 1))
            self.assertTrue(torch.all(v_batch >= -1.0) and torch.all(v_batch <= 1.0))

    def test_predict_board_correctness(self):
        """4. Verify correctness of predict(board, player, device) on both models."""
        board = Board()
        # Place several stones
        board.place_stone(9, 9, Board.BLACK)
        board.place_stone(9, 10, Board.WHITE)
        board.place_stone(10, 9, Board.BLACK)
        board.place_stone(8, 9, Board.WHITE)

        devices = [self.device_cpu]
        if self.cuda_available:
            devices.append(self.device_cuda)

        for dev in devices:
            for name, model in [("Baseline", PolicyValueNet.create_baseline()),
                                ("2x Scaled", PolicyValueNet.create_2x())]:
                model = model.to(dev)
                probs, value = model.predict(board, Board.BLACK, dev)

                self.assertIsInstance(probs, np.ndarray)
                self.assertEqual(probs.shape, (361,))
                self.assertIsInstance(value, float)
                self.assertGreaterEqual(value, -1.0)
                self.assertLessEqual(value, 1.0)

                # Probabilities should sum to 1.0 (approx)
                self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

                # Occupied positions should strictly have probability 0.0
                for r, c in [(9, 9), (9, 10), (10, 9), (8, 9)]:
                    idx = r * 19 + c
                    self.assertEqual(probs[idx], 0.0, f"Occupied pos ({r}, {c}) must have prob 0")

                # Legal unoccupied positions can have non-zero probability
                self.assertTrue(np.all(probs >= 0.0))

    def test_checkpoint_serialization_and_from_config(self):
        """5. Verify checkpoint serialization and from_config model reconstruction."""
        model_2x = PolicyValueNet.create_2x()
        dummy_input = torch.randn(2, 7, 19, 19)

        model_2x.eval()
        with torch.no_grad():
            orig_p, orig_v = model_2x(dummy_input)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name

        try:
            # Save config and state dict
            torch.save({
                "iteration": 42,
                "config": model_2x.get_config(),
                "model_state_dict": model_2x.state_dict(),
            }, temp_path)

            # Load checkpoint
            ckpt = torch.load(temp_path, map_location="cpu")
            loaded_config = ckpt["config"]
            reconstructed_model = PolicyValueNet.from_config(loaded_config)
            reconstructed_model.load_state_dict(ckpt["model_state_dict"])
            reconstructed_model.eval()

            self.assertEqual(reconstructed_model.num_channels, 128)
            self.assertEqual(reconstructed_model.num_res_blocks, 10)
            self.assertEqual(reconstructed_model.get_config(), model_2x.get_config())

            with torch.no_grad():
                new_p, new_v = reconstructed_model(dummy_input)

            self.assertTrue(torch.allclose(orig_p, new_p, atol=1e-6))
            self.assertTrue(torch.allclose(orig_v, new_v, atol=1e-6))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_backwards_compatibility_with_existing_checkpoint(self):
        """Verify seamless loading of existing rl_model_best.pt checkpoint."""
        ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "models", "checkpoints", "rl_model_best.pt")
        if not os.path.exists(ckpt_path):
            self.skipTest(f"Existing checkpoint {ckpt_path} not found")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        config = ckpt.get("config", {})

        # Should load via from_config seamlessly
        model = PolicyValueNet.from_config(config)
        st = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(st)
        model.eval()

        # Check inference works
        dummy_input = torch.randn(1, 7, 19, 19)
        with torch.no_grad():
            p, v = model(dummy_input)
        self.assertEqual(p.shape, (1, 361))
        self.assertEqual(v.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
