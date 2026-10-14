import subprocess
import time

CODER_BASE = [
    'coder',
    '--url', 'https://computeshare-access.tail641d9d.ts.net',
    '--use-keyring=false',
    '--no-version-warning',
    'ssh',
    '--disable-autostart'
]

def run_cmd(workspace, cmd_str):
    c = CODER_BASE + [workspace, cmd_str]
    res = subprocess.run(c, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return res.stdout.strip(), res.stderr.strip()

print("[1] Installing numba on ai-lab...")
out, err = run_cmd('owner/ai-lab', '/home/coder/venv/bin/pip install numba')
print("pip output:", out[-200:] if out else err)

print("[2] Streaming omok_worker codebase from jwj-01 to ai-lab...")
p1 = subprocess.Popen(
    CODER_BASE + ['owner/jwj-01', 'cd ~/omok_worker && tar -czf - engine models mcts remote_worker.py'],
    stdout=subprocess.PIPE
)
p2 = subprocess.Popen(
    CODER_BASE + ['owner/ai-lab', 'mkdir -p ~/omok_worker && tar -xzf - -C ~/omok_worker'],
    stdin=p1.stdout,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
p1.stdout.close()
out2, err2 = p2.communicate()
print("Stream finished:", err2.decode(errors='replace') if err2 else "Success")

print("[3] Verifying files on ai-lab...")
out, err = run_cmd('owner/ai-lab', 'ls -la ~/omok_worker && ls -lh ~/omok_worker/models/checkpoints/')
print(out)

print("[4] Quick test run of remote_worker.py on ai-lab (1 game)...")
t0 = time.time()
out, err = run_cmd('owner/ai-lab', 'cd ~/omok_worker && /home/coder/venv/bin/python3 remote_worker.py --games 1 --sims 40 --device cuda --out /tmp/test_ailab.npz')
print(f"Test run completed in {time.time()-t0:.2f}s:")
print(out)
if err:
    print("STDERR:", err)
