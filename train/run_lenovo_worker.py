"""
Lenovo Dedicated Self-Play Worker: Continuously generates batches of games on Intel i5-13420H.
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train.remote_worker import run_remote_worker

def main():
    out_dir = os.path.join(PROJECT_ROOT, "data", "lenovo_stream")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "rl_model_best.pt")

    batch_idx = 1
    # 10 parallel workers on the 12-thread i5-13420H (maximum CPU utilization!)
    num_workers = 10
    games_per_batch = 10
    sims = 40

    print("=" * 60)
    print("      LENOVO DISTRIBUTED SELF-PLAY WORKER DAEMON      ")
    print(f"Workers: {num_workers} | Games/Batch: {games_per_batch} | MCTS Sims: {sims}")
    print("=" * 60)

    while True:
        ts = int(time.time())
        out_file = os.path.join(out_dir, f"worker_lenovo_{batch_idx}_{ts}.npz")
        print(f"\n[Lenovo Batch {batch_idx}] Generating {games_per_batch} games...", flush=True)
        t0 = time.time()
        try:
            run_remote_worker(
                output_file=out_file,
                model_path=model_path,
                num_games=games_per_batch,
                sims=sims,
                num_workers=num_workers
            )
            elapsed = time.time() - t0
            print(f"[Lenovo Batch {batch_idx}] Saved: {out_file} ({elapsed:.1f}s)", flush=True)
        except Exception as e:
            print(f"Error on Lenovo worker batch {batch_idx}: {e}", flush=True)

        batch_idx += 1
        time.sleep(1)

if __name__ == "__main__":
    main()
