"""
Deploy Omok++ worker to gtxgod (RTX 3060 Ti)
"""
import paramiko
import os
import time

def deploy_to_gtxgod():
    host = os.environ["GTXGOD_HOST"]
    username = os.environ["GTXGOD_USER"]
    password = os.environ["GTXGOD_PASSWORD"]
    print(f"[1/4] Connecting to gtxgod ({host})...")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=username, password=password, timeout=5)
    
    print("[2/4] Uploading payload via SFTP...")
    sftp = c.open_sftp()
    try:
        sftp.mkdir('omok_worker')
    except Exception:
        pass
    sftp.put('data/omok_worker_payload.zip', 'omok_worker/payload.zip')
    sftp.close()
    
    print("[3/4] Unpacking payload...")
    cmd = 'powershell -Command "Expand-Archive -Path omok_worker\\payload.zip -DestinationPath omok_worker -Force; Get-ChildItem omok_worker"'
    _, s, e = c.exec_command(cmd)
    print(s.read().decode('cp949', errors='ignore'))
    
    print("[4/4] Setting up uv virtual environment on gtxgod...")
    setup_cmd = 'powershell -Command "cd omok_worker; py -m venv .venv; .\\.venv\\Scripts\\python.exe -m pip install torch numpy numba --index-url https://download.pytorch.org/whl/cu126"'
    _, s2, e2 = c.exec_command(setup_cmd)
    out = s2.read().decode('cp949', errors='ignore')
    print(out)
    
    c.close()
    print("[SUCCESS] Deployment to gtxgod complete!")

if __name__ == '__main__':
    deploy_to_gtxgod()
