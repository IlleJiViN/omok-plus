import subprocess

c = ['coder', '--url', 'https://computeshare-access.tail641d9d.ts.net', '--use-keyring=false', '--no-version-warning', 'ssh', '--disable-autostart', 'owner/jwj-01', '/home/coder/venv/bin/python3 -c "import torch; ckpt = torch.load(\'omok_worker/models/checkpoints/rl_model_best.pt\', map_location=\'cpu\'); print(\'Champion Iteration:\', ckpt.get(\'iteration\'))"']
res = subprocess.run(c, capture_output=True, text=True)
print(res.stdout, res.stderr)
