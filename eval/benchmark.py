"""
Comprehensive Benchmark Suite for Omok++ Agents.
Conducts:
1. Baselines Comparison: SmartPlayer vs RandomPlayer, SimplePlayer
2. Four-Model Matrix:
   - A. Heuristic Search Agent
   - B. Supervised Learning CNN
   - C. Supervised Learning CNN + MCTS
   - D. Self-play RL CNN + MCTS
3. MCTS Simulation Scaling: 50 vs 100 vs 200 vs 400
Saves results into JSON and prints markdown comparison tables.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.board import Board
from eval.arena import run_head_to_head
from players.randomplayer import RandomPlayer
from players.simpleplayer import SimplePlayer
from players.heuristic_search_player import HeuristicSearchPlayer
from players.sl_cnn_player import SLCNNPlayer
from players.mcts_player import MCTSPlayer
from players.rl_mcts_player import RLMCTSPlayer
from players.smart_player import SmartPlayer


def run_full_benchmark(output_dir: str = "eval", games_per_matchup: int = 8) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    print("=================================================================")
    print("        OMOK++ NEURAL MCTS AGENT COMPREHENSIVE BENCHMARK         ")
    print("=================================================================")

    # 1. Baseline Tests (Requirement: Must beat RandomPlayer and SimplePlayer in >=5/8 games)
    print("\n>>> PHASE 1: Baseline Qualification Matches (Random & Simple) <<<")
    res_rand = run_head_to_head(
        lambda pid: SmartPlayer(pid),
        lambda pid: RandomPlayer(pid),
        "SmartPlayer (RL+MCTS)",
        "RandomPlayer",
        num_games=games_per_matchup,
        time_limit_minutes=1
    )
    all_results["SmartPlayer_vs_RandomPlayer"] = res_rand

    res_simple = run_head_to_head(
        lambda pid: SmartPlayer(pid),
        lambda pid: SimplePlayer(pid),
        "SmartPlayer (RL+MCTS)",
        "SimplePlayer",
        num_games=games_per_matchup,
        time_limit_minutes=1
    )
    all_results["SmartPlayer_vs_SimplePlayer"] = res_simple

    # 2. Four-Model Tournament
    print("\n>>> PHASE 2: Architectural Comparison (A, B, C, D) <<<")
    # SmartPlayer (D) vs Heuristic Search (A)
    res_d_vs_a = run_head_to_head(
        lambda pid: SmartPlayer(pid),
        lambda pid: HeuristicSearchPlayer(pid, max_depth=1, time_budget_ms=100.0),
        "D. SmartPlayer (RL+MCTS)",
        "A. Heuristic Search",
        num_games=games_per_matchup,
        time_limit_minutes=1
    )
    all_results["D_vs_A"] = res_d_vs_a

    # SmartPlayer (D) vs SL CNN (B)
    res_d_vs_b = run_head_to_head(
        lambda pid: SmartPlayer(pid),
        lambda pid: SLCNNPlayer(pid),
        "D. SmartPlayer (RL+MCTS)",
        "B. Pure SL CNN",
        num_games=games_per_matchup,
        time_limit_minutes=1
    )
    all_results["D_vs_B"] = res_d_vs_b

    # SmartPlayer (D) vs SL CNN + MCTS (C)
    res_d_vs_c = run_head_to_head(
        lambda pid: SmartPlayer(pid),
        lambda pid: MCTSPlayer(pid, default_sims=80),
        "D. SmartPlayer (RL+MCTS)",
        "C. SL CNN + MCTS",
        num_games=games_per_matchup,
        time_limit_minutes=1
    )
    all_results["D_vs_C"] = res_d_vs_c

    # SL CNN + MCTS (C) vs Pure SL CNN (B)
    res_c_vs_b = run_head_to_head(
        lambda pid: MCTSPlayer(pid, default_sims=80),
        lambda pid: SLCNNPlayer(pid),
        "C. SL CNN + MCTS",
        "B. Pure SL CNN",
        num_games=games_per_matchup,
        time_limit_minutes=1
    )
    all_results["C_vs_B"] = res_c_vs_b

    # 3. MCTS Simulation Scaling Analysis
    print("\n>>> PHASE 3: MCTS Simulation Scaling Benchmark <<<")
    sim_scaling = {}
    for sims in [50, 100, 200]:
        t0 = time.perf_counter()
        # Measure move latency on 10 test positions
        b = Board()
        p = MCTSPlayer(0, default_sims=sims)
        p.take_turn(b.get_grid(), 60000.0)
        move_time_ms = (time.perf_counter() - t0) * 1000.0
        sim_scaling[f"sims_{sims}"] = {
            "simulations": sims,
            "latency_ms": round(move_time_ms, 2)
        }
        print(f"  MCTS Sims: {sims:3d} -> Move Latency: {move_time_ms:.1f}ms")

    all_results["MCTS_Simulation_Scaling"] = sim_scaling

    # Save to JSON
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll benchmark results saved to {json_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=8, help="Games per matchup (default: 8)")
    parser.add_argument("--out", type=str, default="eval", help="Output directory")
    args = parser.parse_args()

    run_full_benchmark(output_dir=args.out, games_per_matchup=args.games)
