"""
Deploy and launch worker daemon on Lenovo laptop via Paramiko SFTP/SSH.
"""

import os
import paramiko

LENOVO_IP = "192.0.2.1"
LENOVO_USER = "illejivin"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# Use authorized ed25519 key
key_path = os.path.expanduser("~/.ssh/id_ed25519")
ssh.connect(LENOVO_IP, username=LENOVO_USER, key_filename=key_path, timeout=10)

sftp = ssh.open_sftp()
# Create remote train directory if not exists
try:
    sftp.mkdir("/home/illejivin/omok_project/train")
except IOError:
    pass

# Upload remote_worker.py and run_lenovo_worker.py
sftp.put(os.path.join(PROJECT_ROOT, "train", "remote_worker.py"), "/home/illejivin/omok_project/train/remote_worker.py")
sftp.put(os.path.join(PROJECT_ROOT, "train", "run_lenovo_worker.py"), "/home/illejivin/omok_project/train/run_lenovo_worker.py")
sftp.close()
print("Files successfully uploaded to Lenovo!")

# Kill any existing worker and launch new background daemon
cmd = (
    "pkill -f run_lenovo_worker.py 2>/dev/null || true; "
    "mkdir -p ~/omok_project/data/lenovo_stream; "
    "cd ~/omok_project && nohup ~/omok_project/venv/bin/python3 ~/omok_project/train/run_lenovo_worker.py > ~/omok_project/worker_lenovo.log 2>&1 & "
    "sleep 2 && ps aux | grep run_lenovo_worker.py"
)
stdin, stdout, stderr = ssh.exec_command(cmd)
print("\n>>> Lenovo Worker Process Status:")
print(stdout.read().decode())
ssh.close()
