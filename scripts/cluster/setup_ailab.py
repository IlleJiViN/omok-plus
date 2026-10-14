import subprocess

def run_remote(workspace, bash_cmd):
    cmd = [
        'coder',
        '--url', 'https://computeshare-access.tail641d9d.ts.net',
        '--use-keyring=false',
        '--no-version-warning',
        'ssh',
        '--disable-autostart',
        workspace,
        bash_cmd
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return res.stdout.strip(), res.stderr.strip()

# Check if numba and torch are in /home/coder/venv on ai-lab
out, err = run_remote('owner/ai-lab', '/home/coder/venv/bin/python3 -c "import torch, numpy; print(\'Torch & Numpy OK\')"')
print("ai-lab packages:", out)

# Check git/rsync or transfer
out, err = run_remote('owner/jwj-01', 'cd ~/omok_worker && tar -czf /tmp/omok_worker.tar.gz engine models mcts train data/data_generator.py remote_worker.py models/checkpoints/rl_model_best.pt && ls -lh /tmp/omok_worker.tar.gz')
print("jwj-01 tar:", out)
