"""
Remote Self-Play Worker Script for Distributed Omok++ AlphaZero.
Run this script on any other PC, laptop, or server (CPU or GPU) to generate
high-quality Omok++ self-play game datasets for centralized training.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

_curr_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_curr_dir) == "train":
    _project_root = os.path.dirname(_curr_dir)
else:
    _project_root = _curr_dir
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from engine.board import Board
from models.policy_value_net import PolicyValueNet
from mcts.mcts import MCTS
from data.data_generator import augment_8fold


def run_remote_worker(output_file: str,
                      model_path: str = "models/checkpoints/rl_model_best.pt",
                      num_games: int = 200,
                      sims: int = 40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("      DISTRIBUTED OMOK++ SELF-PLAY WORKER CLIENT      ")
    print("=" * 65)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Standard CPU'})")
    print(f"Target Games: {num_games} games")
    print(f"MCTS Simulations: {sims} sims / move")
    print(f"Output File: {output_file}")
    print("-" * 65)

    if not os.path.exists(model_path):
        alt_path = os.path.join(_project_root, model_path)
        if os.path.exists(alt_path):
            model_path = alt_path

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        if config:
            model = PolicyValueNet.from_config(config).to(device)
        else:
            model = PolicyValueNet.create_baseline().to(device)
        st = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(st)
        print(f"Loaded weights from {model_path} (architecture: {model.get_config()})")
    else:
        model = PolicyValueNet.create_baseline().to(device)
        print("Using base initialization.")
    model.eval()

def _worker_process(wid: int, num_games: int, sims: int, model_weights: dict, config: dict, device_str: str, out_q):
    try:
        import torch
        if device_str == "cpu":
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        dev = torch.device(device_str)
        if config:
            model = PolicyValueNet.from_config(config).to(dev)
        else:
            model = PolicyValueNet.create_baseline().to(dev)
        model.load_state_dict(model_weights)
        model.eval()

        samples = []
        for _ in range(num_games):
            board = Board()
            mcts = MCTS(model, dev, c_puct=1.5, n_simulations=sims, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
            curr = Board.BLACK
            step = 0
            history = []

            while True:
                legal = board.get_legal_moves()
                if not legal:
                    winner = -1
                    break

                probs = mcts.search(board, curr, add_noise=True, n_sims=sims)
                if step < 6:
                    p_sum = np.sum(probs)
                    act = int(np.random.choice(361, p=probs / p_sum)) if p_sum > 0 else int(np.argmax(probs))
                else:
                    act = int(np.argmax(probs))

                history.append((board.copy(), act, curr))
                board.place_stone(act // 19, act % 19, curr)
                mcts.step_root(act)
                step += 1

                over, winner = board.check_game_status()
                if over:
                    break
                curr = 1 - curr

            for b, act, p in history:
                st_tensor = PolicyValueNet.encode_board(b, p)
                val = 1.0 if winner == p else (-1.0 if winner == 1 - p else 0.0)
                for aug_s, aug_a in augment_8fold(st_tensor, act):
                    samples.append((aug_s, aug_a, val))

        out_q.put((wid, samples))
    except Exception as e:
        import traceback
        traceback.print_exc()
        out_q.put((wid, []))


def run_remote_worker(output_file: str,
                      model_path: str = "models/checkpoints/rl_model_best.pt",
                      num_games: int = 100,
                      sims: int = 40,
                      num_workers: int = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("      DISTRIBUTED OMOK++ SELF-PLAY WORKER CLIENT      ")
    print("=" * 65)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Standard CPU'})")
    print(f"Target Games: {num_games} games across {num_workers} worker processes")
    print(f"MCTS Simulations: {sims} sims / move")
    print(f"Output File: {output_file}")
    print("-" * 65)

    if not os.path.exists(model_path):
        alt_path = os.path.join(_project_root, model_path)
        if os.path.exists(alt_path):
            model_path = alt_path

    config = {}
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location="cpu")
        config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        st = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        print(f"Loaded weights from {model_path} (architecture: {config})")
    else:
        dummy = PolicyValueNet.create_baseline()
        st = dummy.state_dict()
        print("Using base initialization.")

    all_samples = []
    t_start = time.time()

    if num_workers <= 1:
        # Sequential execution
        import queue
        q = queue.Queue()
        _worker_process(0, num_games, sims, st, config, str(device), q)
        _, all_samples = q.get()
    else:
        # Parallel multiprocessing
        try:
            torch.multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        import torch.multiprocessing as mp
        ctx = mp.get_context("spawn")
        q = ctx.Queue()

        games_per = num_games // num_workers
        rem = num_games % num_workers
        procs = []

        for wid in range(num_workers):
            w_games = games_per + (1 if wid < rem else 0)
            if w_games <= 0:
                continue
            p = ctx.Process(target=_worker_process, args=(wid, w_games, sims, st, config, str(device), q))
            p.start()
            procs.append(p)

        finished = 0
        while finished < len(procs):
            wid, w_samples = q.get()
            all_samples.extend(w_samples)
            finished += 1
            elapsed = time.time() - t_start
            print(f"  [Worker {wid+1}/{len(procs)}] Completed | Collected {len(all_samples):,} total samples so far ({elapsed:.1f}s)", flush=True)

        for p in procs:
            p.join()

    # Save to disk
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    states = np.array([s[0] for s in all_samples], dtype=np.float32)
    actions = np.array([s[1] for s in all_samples], dtype=np.int64)
    values = np.array([s[2] for s in all_samples], dtype=np.float32)
    np.savez_compressed(output_file, states=states, actions=actions, values=values)

    total_time = time.time() - t_start
    print("=" * 65)
    print(f"[COMPLETE] Generated {num_games} games ({len(all_samples):,} augmented samples) in {total_time:.1f}s ({num_games/(max(0.01, total_time))*60:.1f} games/min)!")
    print(f"Saved to: {os.path.abspath(output_file)}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/distributed/worker_data.npz", help="Output .npz path")
    parser.add_argument("--model", type=str, default="models/checkpoints/rl_model_best.pt", help="Path to model weights")
    parser.add_argument("--games", type=int, default=100, help="Number of games to generate")
    parser.add_argument("--sims", type=int, default=40, help="MCTS simulations per move")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes")
    args = parser.parse_args()

    run_remote_worker(output_file=args.out, model_path=args.model, num_games=args.games, sims=args.sims, num_workers=args.workers)

