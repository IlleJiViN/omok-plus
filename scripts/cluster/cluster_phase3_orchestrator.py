"""
Cluster Phase 3 Multi-Machine Self-Play & Reinforcement Learning Orchestrator.
Coordinates:
1. Master Trainer: owner/jwj-01 (RTX 5090)
2. Dedicated GPU Worker: owner/ai-lab (RTX 5090)
3. Dedicated CPU Worker: Lenovo Laptop (192.0.2.1)
4. Local GPU Worker: NVIDIA GeForce RTX 4060 Ti
"""

import os
import sys
import time
import subprocess
import threading

CODER_BASE = [
    'coder',
    '--url', 'https://computeshare-access.tail641d9d.ts.net',
    '--use-keyring=false',
    '--no-version-warning',
    'ssh',
    '--disable-autostart'
]

def run_remote(workspace: str, bash_cmd: str):
    cmd = CODER_BASE + [workspace, bash_cmd]
    return subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

def launch_ailab_generator(total_batches=20, games_per_batch=10, sims=60):
    """Continuously generates self-play games on ai-lab (RTX 5090) and pushes to jwj-01 incoming/"""
    print(f"[AI-LAB WORKER] Starting {total_batches} batches of {games_per_batch} games on RTX 5090...")
    for b in range(1, total_batches + 1):
        tag = f"ailab_b{b}_{int(time.time())}"
        out_file = f"/tmp/{tag}.npz"
        cmd = f"cd ~/omok_worker && /home/coder/venv/bin/python3 remote_worker.py --games {games_per_batch} --sims {sims} --workers 12 --out {out_file}"
        res = run_remote('owner/ai-lab', cmd)

        # Stream file to jwj-01 incoming
        p1 = subprocess.Popen(CODER_BASE + ['owner/ai-lab', f'cat {out_file}'], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(CODER_BASE + ['owner/jwj-01', f'cat > ~/omok_worker/data/incoming/{tag}.npz'], stdin=p1.stdout)
        p1.stdout.close()
        p2.communicate()

        # Clean tmp
        run_remote('owner/ai-lab', f'rm -f {out_file}')
        print(f"  [AI-LAB WORKER] Batch {b}/{total_batches} ({games_per_batch} games) pushed to jwj-01 incoming!", flush=True)

if __name__ == "__main__":
    print("Testing cluster connectivity before launch...")
    res_jwj = run_remote('owner/jwj-01', 'nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader')
    res_ai = run_remote('owner/ai-lab', 'nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader')
    print("jwj-01 GPU:", res_jwj.stdout.strip())
    print("ai-lab GPU:", res_ai.stdout.strip())
