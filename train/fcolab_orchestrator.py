"""
FColab Orchestrator for Distributed Omok++ AlphaZero Self-Play.
Provides automated headless dispatching of self-play worker jobs to Google Colab GPUs.

Features:
- Windows termios/tty compatibility patch for google-colab-cli.
- Worker payload packaging (engine, models, MCTS, data, checkpoints).
- Context-managed execution via FColab with automatic cleanup.
- Full support for Dry-Run mode for offline testing without active Colab credentials.
- Command-line interface for dispatching workers.
"""

import os
import sys
import zipfile
import argparse
from typing import Optional, List, Tuple
import numpy as np

# Ensure safe utf-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Apply Windows termios and tty mock before importing fcolab or colab_cli
if os.name == "nt":
    import unittest.mock
    sys.modules.setdefault("termios", unittest.mock.MagicMock())
    sys.modules.setdefault("tty", unittest.mock.MagicMock())

# Resolve paths
_orchestrator_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_orchestrator_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_fcolab_dir = os.path.join(_project_root, "fcolab")
if os.path.exists(_fcolab_dir) and _fcolab_dir not in sys.path:
    sys.path.insert(0, _fcolab_dir)

try:
    from fcolab import FColab
except ImportError:
    # Fallback definition if fcolab package path varies
    from fcolab.core import FColab


class MockFColab(FColab):
    """
    Simulated FColab context manager for dry-run testing and offline validation.
    Mirrors all FColab methods without calling external subprocesses.
    """
    def __init__(self, session_name: str = "default", gpu: str = "T4"):
        super().__init__(session_name=session_name, gpu=gpu)
        self.uploaded_files: List[Tuple[str, str]] = []
        self.executed_commands: List[str] = []
        self.downloaded_files: List[Tuple[str, str]] = []

    def start(self):
        print(f"[MockFColab] [START] Simulated session '{self.session_name}' with GPU {self.gpu}")

    def upload(self, local_path: str, remote_path: str):
        print(f"[MockFColab] [UPLOAD] {local_path} -> {remote_path}")
        self.uploaded_files.append((local_path, remote_path))

    def exec_script(self, local_script_path: str):
        print(f"[MockFColab] [EXEC_SCRIPT] {local_script_path}")
        self.executed_commands.append(f"exec_script: {local_script_path}")

    def execute_remote(self, command: str):
        print(f"[MockFColab] [EXEC_REMOTE] {command}")
        self.executed_commands.append(command)

    def download(self, remote_path: str, local_path: str):
        print(f"[MockFColab] [DOWNLOAD] {remote_path} -> {local_path}")
        self.downloaded_files.append((remote_path, local_path))
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        # Create a valid mock worker dataset if file does not exist
        if not os.path.exists(local_path):
            states = np.zeros((16, 7, 19, 19), dtype=np.float32)
            actions = np.zeros(16, dtype=np.int64)
            values = np.zeros(16, dtype=np.float32)
            np.savez_compressed(local_path, states=states, actions=actions, values=values)
            print(f"   [MockFColab] Generated mock worker output dataset at {local_path}")

    def stop(self):
        print(f"[MockFColab] [STOP] Simulated session '{self.session_name}'")


class FColabOrchestrator:
    """
    Orchestrates distributed self-play workers on Google Colab via FColab.
    """
    def __init__(self, project_root: Optional[str] = None):
        if project_root is None:
            self.project_root = _project_root
        else:
            self.project_root = os.path.abspath(project_root)

    def create_worker_payload(self, zip_path: str = "data/omok_worker_payload.zip") -> str:
        """
        Zips essential source files:
        - engine/ (board, evaluator, zobrist)
        - models/ (policy_value_net.py, __init__.py)
        - models/checkpoints/rl_model_best.pt (if present)
        - mcts/ (mcts, node, batched_mcts)
        - data/ (data_generator.py)
        - train/ (remote_worker.py)
        - remote_worker.py (at root of zip for direct execution)
        """
        if not os.path.isabs(zip_path):
            zip_path = os.path.join(self.project_root, zip_path)

        os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)
        print(f"[FColabOrchestrator] Packaging worker payload to: {zip_path}")
        file_count = 0

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # 1. engine/
            engine_dir = os.path.join(self.project_root, "engine")
            if os.path.exists(engine_dir):
                for root, dirs, files in os.walk(engine_dir):
                    if "__pycache__" in dirs:
                        dirs.remove("__pycache__")
                    for f in files:
                        if f.endswith((".py", ".pyd", ".so")):
                            full_p = os.path.join(root, f)
                            rel_p = os.path.relpath(full_p, self.project_root)
                            zf.write(full_p, rel_p)
                            file_count += 1

            # 2. models/ (source files)
            models_dir = os.path.join(self.project_root, "models")
            for f in ["__init__.py", "policy_value_net.py"]:
                full_p = os.path.join(models_dir, f)
                if os.path.exists(full_p):
                    zf.write(full_p, os.path.join("models", f))
                    file_count += 1

            # 3. Champion checkpoint: models/checkpoints/rl_model_best.pt
            best_ckpt = os.path.join(models_dir, "checkpoints", "rl_model_best.pt")
            if os.path.exists(best_ckpt):
                zf.write(best_ckpt, os.path.join("models", "checkpoints", "rl_model_best.pt"))
                file_count += 1
                size_mb = os.path.getsize(best_ckpt) / (1024 * 1024)
                print(f"   + Bundled champion checkpoint: rl_model_best.pt ({size_mb:.2f} MB)")
            else:
                print(f"   ! Note: {best_ckpt} not found, omitting from payload.")

            # 4. mcts/
            mcts_dir = os.path.join(self.project_root, "mcts")
            if os.path.exists(mcts_dir):
                for root, dirs, files in os.walk(mcts_dir):
                    if "__pycache__" in dirs:
                        dirs.remove("__pycache__")
                    for f in files:
                        if f.endswith((".py", ".pyd", ".so")):
                            full_p = os.path.join(root, f)
                            rel_p = os.path.relpath(full_p, self.project_root)
                            zf.write(full_p, rel_p)
                            file_count += 1

            # 5. data/
            data_dir = os.path.join(self.project_root, "data")
            for f in ["__init__.py", "data_generator.py"]:
                full_p = os.path.join(data_dir, f)
                if os.path.exists(full_p):
                    zf.write(full_p, os.path.join("data", f))
                    file_count += 1

            # 6. train/ (remote_worker.py)
            train_dir = os.path.join(self.project_root, "train")
            for f in ["__init__.py", "remote_worker.py"]:
                full_p = os.path.join(train_dir, f)
                if os.path.exists(full_p):
                    zf.write(full_p, os.path.join("train", f))
                    file_count += 1

            # Add root remote_worker.py so 'python3 /content/remote_worker.py' works
            worker_src = os.path.join(train_dir, "remote_worker.py")
            if os.path.exists(worker_src):
                zf.write(worker_src, "remote_worker.py")
                file_count += 1

        zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"[FColabOrchestrator] Payload created: {zip_path} ({file_count} files, {zip_size_mb:.2f} MB)")
        return zip_path

    def dispatch_colab_worker(self,
                              session_name: str = "omok_worker_1",
                              gpu: str = "T4",
                              num_games: int = 100,
                              sims: int = 40,
                              dry_run: bool = False,
                              payload_path: Optional[str] = None) -> str:
        """
        Dispatches a self-play worker to Google Colab using FColab.
        """
        if payload_path is None or not os.path.exists(payload_path):
            payload_path = self.create_worker_payload()

        local_out_dir = os.path.join(self.project_root, "data", "distributed")
        os.makedirs(local_out_dir, exist_ok=True)
        local_out_path = os.path.join(local_out_dir, f"{session_name}.npz")

        print("=" * 70)
        print("          FCOLAB DISTRIBUTED SELF-PLAY WORKER DISPATCHER          ")
        print("=" * 70)
        print(f"Session Name: {session_name}")
        print(f"Remote GPU:   {gpu}")
        print(f"Target Games: {num_games} games")
        print(f"MCTS Sims:    {sims} sims / move")
        print(f"Payload Path: {payload_path}")
        print(f"Output File:  {local_out_path}")
        print(f"Dry Run Mode: {dry_run}")
        print("-" * 70)

        colab_cls = MockFColab if dry_run else FColab

        # Execute remote workflow inside FColab context manager
        with colab_cls(session_name=session_name, gpu=gpu) as colab:
            # 1. Upload payload
            remote_payload = "/content/omok_worker_payload.zip"
            colab.upload(payload_path, remote_payload)

            # 2. Unzip and install dependencies (torch, numba, numpy)
            unzip_cmd = f"unzip -o {remote_payload} -d /content && pip install -q torch numba numpy"
            colab.execute_remote(unzip_cmd)

            # 3. Execute remote worker self-play
            worker_cmd = f"python3 /content/remote_worker.py --games {num_games} --sims {sims} --out /content/worker_out.npz"
            colab.execute_remote(worker_cmd)

            # 4. Download generated self-play dataset
            colab.download("/content/worker_out.npz", local_out_path)

        print("-" * 70)
        print(f"Worker dispatch finished. Data saved to: {local_out_path}")
        print("=" * 70)
        return local_out_path


def main():
    parser = argparse.ArgumentParser(description="FColab Distributed Omok++ Self-Play Orchestrator")
    parser.add_argument("--games", type=int, default=100, help="Number of self-play games to generate")
    parser.add_argument("--sims", type=int, default=40, help="Number of MCTS simulations per move")
    parser.add_argument("--gpu", type=str, default="T4", help="Colab GPU type (e.g. T4, V100, A100)")
    parser.add_argument("--session", type=str, default="omok_worker_1", help="Colab session name")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Run in dry-run mode without external Colab CLI")
    parser.add_argument("--payload", type=str, default=None, help="Worker payload zip path")
    args = parser.parse_args()

    orchestrator = FColabOrchestrator()
    orchestrator.dispatch_colab_worker(
        session_name=args.session,
        gpu=args.gpu,
        num_games=args.games,
        sims=args.sims,
        dry_run=args.dry_run,
        payload_path=args.payload
    )


if __name__ == "__main__":
    main()
