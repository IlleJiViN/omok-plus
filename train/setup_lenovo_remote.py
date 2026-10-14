"""
Sets up Lenovo worker node remotely via paramiko SSH.
"""

import os
import paramiko

LENOVO_IP = os.environ["LENOVO_HOST"]
LENOVO_USER = os.environ["LENOVO_USER"]
LENOVO_PASS = os.environ["LENOVO_PASSWORD"]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(LENOVO_IP, username=LENOVO_USER, password=LENOVO_PASS, timeout=10)

def run_cmd(cmd, desc=""):
    print(f"\n>>> [{desc}] Running: {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    exit_status = stdout.channel.recv_exit_status()
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print(out)
    if err and exit_status != 0:
        print(f"Error (code {exit_status}): {err}")
    return exit_status, out

# 1. Authorize SSH public key
pubkey_path = os.path.expanduser("~/.ssh/id_ed25519.pub")
if os.path.exists(pubkey_path):
    with open(pubkey_path, "r") as f:
        pub = f.read().strip()
    run_cmd(f'mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo "{pub}" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys', "Authorize SSH Key")

# 2. Check Tailscale received files or fetch omok_lenovo.zip
run_cmd("tailscale file get . || true", "Get Taildrop Files")
run_cmd("ls -la", "List Home Directory")

# 3. Check if omok_lenovo.zip exists, if not download from Windows HTTP server
run_cmd("if [ ! -f omok_lenovo.zip ]; then wget -q http://192.0.2.1:8899/omok_lenovo.zip; fi", "Fetch Zip Archive")
run_cmd("ls -lh omok_lenovo.zip", "Verify Zip")

# 4. Unzip project
run_cmd("rm -rf omok_project && unzip -q omok_lenovo.zip -d omok_project && ls -la omok_project", "Unzip Omok Project")

# 5. Check Python 3 and hardware info
run_cmd("lscpu | grep 'Model name\\|CPU(s):'", "Check CPU Specs")
run_cmd("python3 --version || true", "Check Python3")

ssh.close()
print("\n>>> Remote Inspection Complete! <<<")
