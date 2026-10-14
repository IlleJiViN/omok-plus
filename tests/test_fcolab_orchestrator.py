"""
Unit and integration tests for FColab Orchestrator and distributed worker pipeline.
Verifies:
1. Windows termios/tty compatibility patch for google-colab-cli.
2. Worker payload archive packaging and contents integrity.
3. Dry-run orchestration workflow and dataset artifact creation.
"""

import os
import sys
import zipfile
import tempfile
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.fcolab_orchestrator import FColabOrchestrator, MockFColab, FColab


class TestFColabOrchestrator(unittest.TestCase):

    def setUp(self):
        self.orchestrator = FColabOrchestrator()
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_windows_termios_compatibility_patch(self):
        """1. Verify Windows termios/tty mock prevents ModuleNotFoundError on Windows."""
        self.assertIn("termios", sys.modules, "sys.modules must have 'termios' registered")
        self.assertIn("tty", sys.modules, "sys.modules must have 'tty' registered")

        # Verify importing colab_cli submodules does not crash
        try:
            import colab_cli.console
            import colab_cli.cli
            self.assertTrue(True)
        except ModuleNotFoundError as e:
            self.fail(f"Importing colab_cli failed with ModuleNotFoundError: {e}")

        # Verify FColab class is accessible and instantiable
        colab_instance = FColab(session_name="patch_test", gpu="T4")
        self.assertEqual(colab_instance.session_name, "patch_test")
        self.assertEqual(colab_instance.gpu, "T4")

    def test_payload_creation(self):
        """2. Verify create_worker_payload creates valid zip containing all necessary modules."""
        temp_zip = os.path.join(self.test_dir.name, "payload_test.zip")
        created_path = self.orchestrator.create_worker_payload(zip_path=temp_zip)

        self.assertEqual(os.path.abspath(created_path), os.path.abspath(temp_zip))
        self.assertTrue(os.path.exists(temp_zip), "Payload zip must exist on disk")
        self.assertTrue(zipfile.is_zipfile(temp_zip), "Payload must be a valid zip archive")

        with zipfile.ZipFile(temp_zip, "r") as zf:
            namelist = zf.namelist()
            print(f"\n[Payload Test] Packed {len(namelist)} items into {temp_zip}")

            # Check essential source files
            required_files = [
                "engine/board.py",
                "models/policy_value_net.py",
                "mcts/mcts.py",
                "data/data_generator.py",
                "train/remote_worker.py",
                "remote_worker.py",
            ]
            for req in required_files:
                self.assertIn(req, namelist, f"Missing required file in payload: {req}")

            # Check champion checkpoint if it exists in source tree
            src_ckpt = os.path.join(self.orchestrator.project_root, "models", "checkpoints", "rl_model_best.pt")
            if os.path.exists(src_ckpt):
                self.assertIn("models/checkpoints/rl_model_best.pt", namelist)

            # Check that no __pycache__ or .pyc files are included
            for name in namelist:
                self.assertNotIn("__pycache__", name, f"Unwanted cache dir in payload: {name}")
                self.assertFalse(name.endswith(".pyc"), f"Unwanted .pyc file in payload: {name}")

    def test_dry_run_orchestration_loop(self):
        """3. Verify dispatch_colab_worker dry_run workflow and artifact generation."""
        session_name = "test_unit_worker_dry"
        output_file = self.orchestrator.dispatch_colab_worker(
            session_name=session_name,
            gpu="T4",
            num_games=10,
            sims=20,
            dry_run=True
        )

        expected_path = os.path.join(self.orchestrator.project_root, "data", "distributed", f"{session_name}.npz")
        self.assertEqual(os.path.abspath(output_file), os.path.abspath(expected_path))
        self.assertTrue(os.path.exists(output_file), f"Output dataset {output_file} must exist")

        # Load and validate generated dataset
        with np.load(output_file) as data:
            self.assertIn("states", data)
            self.assertIn("actions", data)
            self.assertIn("values", data)

            states = data["states"]
            actions = data["actions"]
            values = data["values"]

            self.assertEqual(states.dtype, np.float32)
            self.assertEqual(len(states.shape), 4)
            self.assertEqual(states.shape[1:], (7, 19, 19))
            self.assertEqual(actions.dtype, np.int64)
            self.assertEqual(values.dtype, np.float32)
            self.assertEqual(len(states), len(actions))
            self.assertEqual(len(actions), len(values))

        # Clean up generated test output
        if os.path.exists(output_file):
            os.remove(output_file)

    def test_mock_fcolab_context_manager(self):
        """Verify MockFColab records operations and satisfies context manager contract."""
        session = "mock_test_session"
        with MockFColab(session_name=session, gpu="A100") as colab:
            self.assertEqual(colab.session_name, session)
            self.assertEqual(colab.gpu, "A100")
            colab.upload("dummy_local.txt", "/content/dummy_remote.txt")
            colab.execute_remote("echo Hello")
            temp_out = os.path.join(self.test_dir.name, "out.npz")
            colab.download("/content/out.npz", temp_out)

        self.assertEqual(len(colab.uploaded_files), 1)
        self.assertEqual(len(colab.executed_commands), 1)
        self.assertEqual(len(colab.downloaded_files), 1)
        self.assertTrue(os.path.exists(temp_out))


if __name__ == "__main__":
    unittest.main()
