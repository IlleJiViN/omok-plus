import subprocess

def run_jwj(cmd):
    c = ['coder', '--url', 'https://computeshare-access.tail641d9d.ts.net', '--use-keyring=false', '--no-version-warning', 'ssh', '--disable-autostart', 'owner/jwj-01', cmd]
    return subprocess.run(c, capture_output=True, text=True).stdout

print(run_jwj("grep -n -C 5 'EXTERNAL' omok_worker/train/train_large_scale.py"))
