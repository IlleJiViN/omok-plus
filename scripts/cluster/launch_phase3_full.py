import os
import sys
import time
import subprocess
import paramiko

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

print("=" * 65)
print("     LAUNCHING PHASE 3 DISTRIBUTED OLYMPIC OMOK++ TRAINING     ")
print("=" * 65)

# 1. Clean old incoming in jwj-01 if any leftover
run_remote('owner/jwj-01', 'mkdir -p ~/omok_worker/data/incoming')

# 2. Start Master Trainer on owner/jwj-01 (RTX 5090)
print("\n[1/3] Starting Master Trainer on owner/jwj-01 (RTX 5090)...")
train_cmd = "nohup /home/coder/venv/bin/python3 omok_worker/train/train_large_scale_rl.py --total_games 1200 --iters 10 --workers 24 --sims 60 --model_type 2x > ~/omok_worker/train_5090_phase3.log 2>&1 &"
run_remote('owner/jwj-01', train_cmd)
time.sleep(2)
res_log = run_remote('owner/jwj-01', 'head -n 20 ~/omok_worker/train_5090_phase3.log')
print(res_log.stdout.strip())

# 3. Start Dedicated Self-Play Worker Loop on owner/ai-lab (RTX 5090)
print("\n[2/3] Setting up background worker streamer on owner/ai-lab (RTX 5090)...")
ailab_script = """import time, subprocess, os
for b in range(1, 31):
    tag = f'ailab_b{b}_{int(time.time())}'
    out = f'/tmp/{tag}.npz'
    subprocess.run(['/home/coder/venv/bin/python3', 'remote_worker.py', '--games', '10', '--sims', '60', '--workers', '12', '--out', out], cwd='/home/coder/omok_worker')
    if os.path.exists(out):
        # Move directly to jwj-01 incoming via /tmp or coder ssh
        c_push = ['coder', '--url', 'https://computeshare-access.tail641d9d.ts.net', '--use-keyring=false', '--no-version-warning', 'ssh', '--disable-autostart', 'owner/jwj-01', f'cat > ~/omok_worker/data/incoming/{tag}.npz']
        with open(out, 'rb') as f:
            subprocess.run(c_push, stdin=f)
        os.remove(out)
    time.sleep(1)
"""
run_remote('owner/ai-lab', f"cat << 'EOF' > ~/omok_worker/stream_worker.py\n{ailab_script}\nEOF")
run_remote('owner/ai-lab', "nohup /home/coder/venv/bin/python3 ~/omok_worker/stream_worker.py > ~/omok_worker/stream_worker.log 2>&1 &")
time.sleep(2)
print("  [OK] ai-lab background worker launched!")

# 4. Start Lenovo Worker (192.0.2.1)
print("\n[3/3] Launching CPU self-play worker on Lenovo laptop (192.0.2.1)...")
try:
    key = paramiko.Ed25519Key.from_private_key_file('C:/Users/pc/.ssh/id_ed25519')
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.0.2.1', username='illejivin', pkey=key, timeout=10)

    # Launch 60 games on Lenovo in background
    cmd_lenovo = "nohup ~/omok_project/venv/bin/python ~/omok_project/train/remote_worker.py --games 30 --sims 40 --workers 4 --out ~/omok_project/lenovo_phase3.npz > ~/omok_project/worker.log 2>&1 &"
    ssh.exec_command(cmd_lenovo)
    ssh.close()
    print("  [OK] Lenovo CPU worker launched successfully!")
except Exception as e:
    print(f"  [!] Lenovo launch notice: {e}")

print("\n" + "=" * 65)
print("  PHASE 3 DUAL RTX 5090 + LENOVO CLUSTER IS OFFICIALLY RUNNING!  ")
print("=" * 65)
