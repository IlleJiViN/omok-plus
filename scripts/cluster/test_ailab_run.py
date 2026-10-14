import subprocess

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

print("Copying data/ from jwj-01 to ai-lab...")
p1 = subprocess.Popen(
    CODER_BASE + ['owner/jwj-01', 'cd ~/omok_worker && tar -czf - data/data_generator.py data/__init__.py'],
    stdout=subprocess.PIPE
)
p2 = subprocess.Popen(
    CODER_BASE + ['owner/ai-lab', 'tar -xzf - -C ~/omok_worker'],
    stdin=p1.stdout,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
p1.stdout.close()
p2.communicate()

print("Running 1 game test on ai-lab (RTX 5090)...")
out, err = run_cmd('owner/ai-lab', 'cd ~/omok_worker && /home/coder/venv/bin/python3 remote_worker.py --games 1 --sims 40 --device cuda --out /tmp/test_ailab.npz')
print("STDOUT:\n", out)
print("STDERR:\n", err)
