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
    return res.stdout.strip()

print("Files in ai-lab /home/coder:")
print(run_remote('owner/ai-lab', 'ls -la /home/coder'))
