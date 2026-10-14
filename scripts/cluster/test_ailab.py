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

print("=== jwj-01 ===")
print("Hostname:", run_remote('owner/jwj-01', 'hostname'))
print("Uptime:", run_remote('owner/jwj-01', 'uptime'))
print("nvidia-smi:", run_remote('owner/jwj-01', 'nvidia-smi --query-gpu=gpu_name,memory.total,memory.used,utilization.gpu --format=csv,noheader'))

print("\n=== ai-lab ===")
print("Hostname:", run_remote('owner/ai-lab', 'hostname'))
print("Uptime:", run_remote('owner/ai-lab', 'uptime'))
print("nvidia-smi:", run_remote('owner/ai-lab', 'nvidia-smi --query-gpu=gpu_name,memory.total,memory.used,utilization.gpu --format=csv,noheader'))
