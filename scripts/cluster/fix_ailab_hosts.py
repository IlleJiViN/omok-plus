import subprocess

c = ['coder', '--url', 'https://computeshare-access.tail641d9d.ts.net', '--use-keyring=false', '--no-version-warning', 'ssh', '--disable-autostart', 'owner/ai-lab', 'echo "192.0.2.1 computeshare-access.tail641d9d.ts.net" | sudo tee -a /etc/hosts']
res = subprocess.run(c, capture_output=True, text=True)
print("hosts output:", res.stdout, res.stderr)
