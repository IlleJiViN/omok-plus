import subprocess

c = ['coder', '--url', 'https://computeshare-access.tail641d9d.ts.net', '--use-keyring=false', '--no-version-warning', 'ssh', '--disable-autostart', 'owner/jwj-01', 'sed -n "200,320p" omok_worker/train/train_large_scale_rl.py']
print(subprocess.run(c, capture_output=True, text=True).stdout)
