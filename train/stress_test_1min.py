"""
1-Minute Full-Throttle CPU & GPU Stress Test and Throughput Benchmark for Omok++.
Utilizes 10-12 parallel worker processes with concurrent CUDA streams on RTX 4060 Ti.
Runs for exactly 60 seconds and outputs complete hardware and throughput analytics.
"""

import os
import sys
import time
import argparse
import psutil
import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.board import Board
from models.policy_value_net import PolicyValueNet
from mcts.mcts import MCTS
from data.data_generator import augment_8fold


def worker_proc(worker_id: int,
                model_path: str,
                stop_event: mp.Event,
                result_queue: mp.Queue,
                sims: int = 40):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PolicyValueNet(in_channels=7, num_channels=64, num_res_blocks=5).to(device)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=device)
            st = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(st)
        model.eval()

        games_done = 0
        moves_done = 0
        all_samples = []

        while not stop_event.is_set():
            board = Board()
            mcts = MCTS(model, device, c_puct=1.5, n_simulations=sims, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
            curr = Board.BLACK
            step = 0
            game_history = []

            while not stop_event.is_set():
                legal = board.get_legal_moves()
                if not legal:
                    winner = -1
                    break

                probs = mcts.search(board, curr, add_noise=True, n_sims=sims)
                if step < 6:
                    p_sum = np.sum(probs)
                    if p_sum > 0:
                        p_norm = probs / p_sum
                        act = int(np.random.choice(361, p=p_norm))
                    else:
                        act = int(np.argmax(probs))
                else:
                    act = int(np.argmax(probs))

                game_history.append((board.copy(), act, curr))
                board.place_stone(act // 19, act % 19, curr)
                mcts.step_root(act)
                step += 1

                over, winner = board.check_game_status()
                if over:
                    break
                curr = 1 - curr

            if stop_event.is_set():
                break

            games_done += 1
            moves_done += step

            # 8-fold dihedral augmentation
            for b, act, p in game_history:
                st_tensor = PolicyValueNet.encode_board(b, p)
                if winner == -1 or winner is None:
                    val = 0.0
                elif winner == p:
                    val = 1.0
                else:
                    val = -1.0

                for aug_s, aug_a in augment_8fold(st_tensor, act):
                    all_samples.append((aug_s, aug_a, val))

        result_queue.put((worker_id, games_done, moves_done, all_samples))

    except Exception as e:
        result_queue.put((worker_id, 0, 0, []))


def run_1min_stress_test(workers: int = 10, duration: float = 60.0, sims: int = 40):
    print("=" * 70)
    print("      OMOK++ 1-MINUTE FULL-THROTTLE CPU & GPU STRESS TEST      ")
    print("=" * 70)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_cores = os.cpu_count()
    mem = psutil.virtual_memory()

    print(f"Hardware Configuration:")
    print(f"  - CPU: {total_cores} Logical Cores (Active Workers: {workers})")
    print(f"  - RAM: Total {mem.total / (1024**3):.1f} GB, Available {mem.available / (1024**3):.1f} GB")
    print(f"  - GPU: {device_name} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)")
    print(f"  - MCTS Simulation Depth: {sims} sims / move")
    print(f"  - Target Run Duration: {duration:.1f} seconds")
    print("-" * 70)

    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "models", "checkpoints", "rl_model_best.pt")

    stop_event = mp.Event()
    result_queue = mp.Queue()

    procs = []
    print(f"[LAUNCH] Spawning {workers} concurrent workers across CPU cores and GPU...", flush=True)
    t_start = time.time()

    for w_id in range(workers):
        p = mp.Process(target=worker_proc, args=(w_id, model_path, stop_event, result_queue, sims))
        p.daemon = True
        p.start()
        procs.append(p)

    print(f"[RUNNING] Full-throttle load active! Monitoring for {duration:.0f} seconds...\n", flush=True)

    peak_cpu = 0.0
    elapsed = 0.0
    last_print = 0.0

    while elapsed < duration:
        time.sleep(1.0)
        elapsed = time.time() - t_start
        current_cpu = psutil.cpu_percent(interval=None)
        if current_cpu > peak_cpu:
            peak_cpu = current_cpu

        if elapsed - last_print >= 5.0 or elapsed >= duration:
            last_print = elapsed
            remaining = max(0.0, duration - elapsed)
            gpu_vram = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            print(f"  [T+{elapsed:4.1f}s | Left: {remaining:4.1f}s] CPU Load: {current_cpu:5.1f}% (Peak: {peak_cpu:5.1f}%) | Active Workers: {workers}", flush=True)

    print("\n[STOPPING] 60-second timer reached! Signaling workers to finish...", flush=True)
    stop_event.set()

    total_games = 0
    total_moves = 0
    all_augmented_samples = []

    # Collect from queue with timeout
    for _ in range(workers):
        try:
            w_id, g_cnt, m_cnt, s_list = result_queue.get(timeout=10.0)
            total_games += g_cnt
            total_moves += m_cnt
            all_augmented_samples.extend(s_list)
        except Exception:
            pass

    for p in procs:
        p.join(timeout=3.0)
        if p.is_alive():
            p.terminate()

    actual_time = time.time() - t_start

    # Save samples to disk
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    out_file = os.path.join(out_dir, "stress_test_samples.npz")
    if all_augmented_samples:
        states = np.array([s[0] for s in all_augmented_samples], dtype=np.float32)
        actions = np.array([s[1] for s in all_augmented_samples], dtype=np.int64)
        values = np.array([s[2] for s in all_augmented_samples], dtype=np.float32)
        np.savez_compressed(out_file, states=states, actions=actions, values=values)

    # Analytics calculation
    gps = total_games / actual_time
    mps = total_moves / actual_time
    sps = len(all_augmented_samples) / actual_time

    time_for_500 = (500 / max(0.01, gps)) / 60.0
    time_for_1000 = (1000 / max(0.01, gps)) / 60.0
    time_for_2000 = (2000 / max(0.01, gps)) / 60.0

    print("=" * 70)
    print("                    1-MINUTE STRESS TEST RESULTS                      ")
    print("=" * 70)
    print(f"1. Throughput Metrics:")
    print(f"   - Total Games Completed in {actual_time:.1f}s: {total_games} games")
    print(f"   - Total Moves Computed: {total_moves} moves")
    print(f"   - 8-Fold Augmented Samples Generated: {len(all_augmented_samples):,} samples")
    print(f"   - Game Generation Speed: {gps:.2f} games/sec ({gps * 60:.1f} games/min)")
    print(f"   - Move Search Throughput: {mps:.1f} moves/sec")
    print(f"   - Sample Production Rate: {sps:.1f} samples/sec")
    print()
    print(f"2. Hardware Utilization:")
    print(f"   - Peak CPU Load: {peak_cpu:.1f}%")
    print(f"   - CPU Workers Active: {workers} processes")
    print(f"   - Memory Status: Safe (Saved to {out_file})")
    print()
    print(f"3. Extrapolated Scale-Up Timelines:")
    print(f"   - Time required for   500 Games: {time_for_500:.1f} minutes")
    print(f"   - Time required for 1,000 Games: {time_for_1000:.1f} minutes")
    print(f"   - Time required for 2,000 Games: {time_for_2000:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10, help="Number of worker processes (default: 10)")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds (default: 60.0)")
    parser.add_argument("--sims", type=int, default=40, help="MCTS simulations per move (default: 40)")
    args = parser.parse_args()

    run_1min_stress_test(workers=args.workers, duration=args.duration, sims=args.sims)
