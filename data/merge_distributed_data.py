"""
Merge Distributed Self-Play Datasets into Replay Buffer for Centralized Training.
Gathers all .npz files from a directory and merges them into a unified dataset.
"""

import os
import sys
import glob
import argparse
import numpy as np


def merge_datasets(input_dir: str = "data/distributed", output_file: str = "data/merged_distributed_dataset.npz"):
    npz_files = glob.glob(os.path.join(input_dir, "*.npz"))
    if not npz_files:
        print(f"[ERROR] No .npz files found in {input_dir}")
        return None

    print(f"Found {len(npz_files)} distributed dataset files in {input_dir}:")
    all_states = []
    all_actions = []
    all_values = []
    total_samples = 0

    for f_path in npz_files:
        try:
            d = np.load(f_path)
            s, a, v = d["states"], d["actions"], d["values"]
            all_states.append(s)
            all_actions.append(a)
            all_values.append(v)
            print(f"  + {os.path.basename(f_path)}: {len(s):,} samples ({os.path.getsize(f_path)/(1024**2):.1f} MB)")
            total_samples += len(s)
        except Exception as e:
            print(f"  - Failed to load {f_path}: {e}")

    merged_s = np.concatenate(all_states, axis=0)
    merged_a = np.concatenate(all_actions, axis=0)
    merged_v = np.concatenate(all_values, axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    np.savez_compressed(output_file, states=merged_s, actions=merged_a, values=merged_v)

    print("-" * 65)
    print(f"[MERGE COMPLETE] Total Merged Samples: {total_samples:,} samples")
    print(f"Output File: {output_file} ({os.path.getsize(output_file)/(1024**2):.1f} MB)")
    print("-" * 65)
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/distributed", help="Directory containing remote .npz files")
    parser.add_argument("--out", type=str, default="data/merged_distributed_dataset.npz", help="Output path for merged dataset")
    args = parser.parse_args()

    merge_datasets(input_dir=args.input_dir, output_file=args.out)
