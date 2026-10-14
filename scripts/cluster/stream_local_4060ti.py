"""
Local RTX 4060 Ti Worker Streamer for Phase 3.
Generates games locally on CUDA and streams directly to jwj-01 incoming/.
"""

import os
import sys
import time
import subprocess

CODER_BASE = [
    'coder',
    '--url', 'https://computeshare-access.tail641d9d.ts.net',
    '--use-keyring=false',
    '--no-version-warning',
    'ssh',
    '--disable-autostart'
]

def stream_local_worker():
    for b in range(1, 31):
        tag = f"local_4060ti_b{b}_{int(time.time())}"
        out = f"data/incoming_local/{tag}.npz"
        os.makedirs("data/incoming_local", exist_ok=True)

        # Run 6 games per batch with 60 sims on RTX 4060 Ti
        cmd = [
            sys.executable,
            "train/remote_worker.py",
            "--games", "6",
            "--sims", "60",
            "--workers", "6",
            "--out", out
        ]
        subprocess.run(cmd)

        if os.path.exists(out):
            c_push = CODER_BASE + ['owner/jwj-01', f'cat > ~/omok_worker/data/incoming/{tag}.npz']
            with open(out, 'rb') as f:
                subprocess.run(c_push, stdin=f)
            os.remove(out)
        time.sleep(1)

if __name__ == "__main__":
    stream_local_worker()
