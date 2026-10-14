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

out, err = run_remote('owner/ai-lab', '/home/coder/venv/bin/python3 -c "import numba; print(\'Numba OK\')"')
print("Numba check:", out, err)
