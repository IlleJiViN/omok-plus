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

# 1. Start Dedicated Self-Play Worker Loop on owner/ai-lab (RTX 5090)
print("[1/2] Launching background worker streamer on owner/ai-lab (RTX 5090)...")
ailab_script = """import time, subprocess, os
for b in range(1, 51):
    tag = f'ailab_b{b}_{int(time.time())}'
    out = f'/tmp/{tag}.npz'
    subprocess.run(['/home/coder/venv/bin/python3', 'remote_worker.py', '--games', '10', '--sims', '60', '--workers', '12', '--out', out], cwd='/home/coder/omok_worker')
    if os.path.exists(out):
        c_push = ['coder', '--url', 'https://computeshare-access.tail641d9d.ts.net', '--use-keyring=false', '--no-version-warning', 'ssh', '--disable-autostart', 'owner/jwj-01', f'cat > ~/omok_worker/data/incoming/{tag}.npz']
        with open(out, 'rb') as f:
            subprocess.run(c_push, stdin=f)
        os.remove(out)
    time.sleep(1)
"""
run_remote('owner/ai-lab', f"cat << 'EOF' > ~/omok_worker/stream_worker.py\n{ailab_script}\nEOF")
run_remote('owner/ai-lab', "nohup /home/coder/venv/bin/python3 ~/omok_worker/stream_worker.py > ~/omok_worker/stream_worker.log 2>&1 &")
time.sleep(2)
print("  [OK] ai-lab background worker streaming games to jwj-01!")

# 2. Start Lenovo Worker (192.0.2.1)
print("\n[2/2] Launching CPU self-play worker on Lenovo laptop (192.0.2.1)...")
try:
    key = paramiko.Ed25519Key.from_private_key_file('C:/Users/pc/.ssh/id_ed25519')
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.0.2.1', username='illejivin', pkey=key, timeout=10)

    cmd_lenovo = "nohup ~/omok_project/venv/bin/python ~/omok_project/train/remote_worker.py --games 30 --sims 40 --workers 4 --out ~/omok_project/lenovo_phase3.npz > ~/omok_project/worker.log 2>&1 &"
    ssh.exec_command(cmd_lenovo)
    ssh.close()
    print("  [OK] Lenovo CPU worker launched successfully!")
except Exception as e:
    print(f"  [!] Lenovo notice: {e}")

print("\nAll cluster nodes successfully engaged!")
