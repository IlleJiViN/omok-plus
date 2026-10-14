import time, subprocess, os

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

ailab_script = """import time, subprocess, os
for b in range(1, 101):
    tag = f'ailab_clean_b{b}_{int(time.time())}'
    out = f'/tmp/{tag}.npz'
    subprocess.run(['/home/coder/venv/bin/python3', 'remote_worker.py', '--games', '8', '--sims', '60', '--workers', '4', '--out', out], cwd='/home/coder/omok_worker')
    if os.path.exists(out):
        c_push = ['coder', '--url', 'https://computeshare-access.tail641d9d.ts.net', '--use-keyring=false', '--no-version-warning', 'ssh', '--disable-autostart', 'owner/jwj-01', f'cat > ~/omok_worker/data/incoming/{tag}.npz']
        with open(out, 'rb') as f:
            subprocess.run(c_push, stdin=f)
        os.remove(out)
    time.sleep(2)
"""

run_remote('owner/ai-lab', f"cat << 'EOF' > ~/omok_worker/stream_worker_clean.py\n{ailab_script}\nEOF")
run_remote('owner/ai-lab', "nohup /home/coder/venv/bin/python3 ~/omok_worker/stream_worker_clean.py > ~/omok_worker/stream_worker_clean.log 2>&1 &")
time.sleep(2)
print("ai-lab clean streamer launched!")
