import subprocess
import hashlib
import os
import torch

CODER_BASE = [
    'coder',
    '--url', 'https://computeshare-access.tail641d9d.ts.net',
    '--use-keyring=false',
    '--no-version-warning',
    'ssh',
    '--disable-autostart',
    'owner/jwj-01'
]

local_dst = 'models/checkpoints/rl_model_best.pt'
backup_dst = 'models/checkpoints/rl_model_iter_18_backup.pt'

if os.path.exists(local_dst) and not os.path.exists(backup_dst):
    import shutil
    shutil.copy2(local_dst, backup_dst)
    print("Backed up previous Iter 18 champion to", backup_dst)

print("Downloading Iteration 23 Champion from jwj-01...")
c = CODER_BASE + ['cat ~/omok_worker/models/checkpoints/rl_model_best.pt']
with open(local_dst, 'wb') as f:
    subprocess.run(c, stdout=f)

size = os.path.getsize(local_dst)
print(f"Downloaded size: {size:,} bytes ({size/1024/1024:.2f} MB)")

# Verify with torch
ckpt = torch.load(local_dst, map_location='cpu')
it = ckpt.get('iteration')
cfg = ckpt.get('config')
num_params = sum(p.numel() for p in ckpt['model_state_dict'].values())

print(f"Loaded successfully!")
print(f"  Iteration: {it}")
print(f"  Architecture Config: {cfg}")
print(f"  Total Parameters: {num_params:,}")
