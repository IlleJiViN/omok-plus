"""
Local Worker Daemon: Generates self-play games and streams them to the 5090 Central Trainer.
"""

import os
import sys
import time
import subprocess
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train.remote_worker import run_remote_worker


def push_to_5090(local_npz_path: str, remote_filename: str) -> bool:
    """Streams local npz file directly into 5090 incoming buffer via coder ssh."""
    cmd = [
        "coder",
        "--url", "https://computeshare-access.tail641d9d.ts.net",
        "--use-keyring=false",
        "--no-version-warning",
        "ssh",
        "--disable-autostart",
        "owner/jwj-01",
        f"cat > ~/omok_worker/data/incoming/{remote_filename}"
    ]
    try:
        with open(local_npz_path, "rb") as f:
            data = f.read()
        res = subprocess.run(cmd, input=data, check=True, capture_output=True)
        print(f"  [STREAM] Successfully pushed {os.path.basename(local_npz_path)} ({len(data)/1024:.1f} KB) to 5090 incoming buffer!")
        return True
    except Exception as e:
        print(f"  [STREAM ERROR] Failed to push to 5090: {e}")
        return False


def sync_lenovo_batches():
    """Vacuums any completed .npz files from Lenovo laptop and pushes to 5090."""
    try:
        import paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        key_path = os.path.expanduser("~/.ssh/id_ed25519")
        ssh.connect("192.0.2.1", username="illejivin", key_filename=key_path, timeout=5)
        sftp = ssh.open_sftp()
        remote_dir = "/home/illejivin/omok_project/data/lenovo_stream"
        try:
            files = sftp.listdir(remote_dir)
        except IOError:
            files = []
        for f in files:
            if f.endswith(".npz"):
                remote_f = f"{remote_dir}/{f}"
                local_tmp = os.path.join(PROJECT_ROOT, "data", "local_stream", f)
                sftp.get(remote_f, local_tmp)
                sftp.remove(remote_f)
                push_to_5090(local_tmp, f)
                if os.path.exists(local_tmp):
                    os.remove(local_tmp)
        sftp.close()
        ssh.close()
    except Exception as e:
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Distributed Omok++ Local Worker Daemon")
    parser.add_argument("--games_per_batch", type=int, default=12, help="Number of games per stream batch")
    parser.add_argument("--sims", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes on this machine")
    parser.add_argument("--continuous", action="store_true", default=True, help="Run continuously in background")
    args = parser.parse_args()

    print("=" * 65)
    print("      OMOK++ LOCAL DISTRIBUTED WORKER DAEMON      ")
    print(f"Games per batch: {args.games_per_batch} | MCTS Sims: {args.sims} | Parallel Workers: {args.workers}")
    print("=" * 65)

    batch_idx = 1
    os.makedirs(os.path.join(PROJECT_ROOT, "data", "local_stream"), exist_ok=True)

    while True:
        ts = int(time.time())
        local_file = os.path.join(PROJECT_ROOT, "data", "local_stream", f"local_batch_{batch_idx}_{ts}.npz")
        remote_name = f"worker_nvidia_{batch_idx}_{ts}.npz"
        model_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "rl_model_best.pt")

        print(f"\n[Batch {batch_idx}] Generating {args.games_per_batch} self-play games (sims={args.sims})...")
        t0 = time.time()
        try:
            run_remote_worker(
                output_file=local_file,
                model_path=model_path,
                num_games=args.games_per_batch,
                sims=args.sims,
                num_workers=args.workers
            )
            elapsed = time.time() - t0

            # Push to 5090 incoming queue
            if os.path.exists(local_file) and os.path.getsize(local_file) > 1000:
                success = push_to_5090(local_file, remote_name)
                if success:
                    os.remove(local_file)

            # Sync and push any batches produced by Lenovo
            sync_lenovo_batches()
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")

        batch_idx += 1
        if not args.continuous:
            break


if __name__ == "__main__":
    main()
