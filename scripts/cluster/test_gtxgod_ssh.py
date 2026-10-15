import os
import paramiko

host = os.environ["GTXGOD_HOST"]
username = os.environ["GTXGOD_USER"]
password = os.environ["GTXGOD_PASSWORD"]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    c.connect(host, username=username, password=password, timeout=5)
    
    cmd1 = 'py -0'
    _, s1, e1 = c.exec_command(cmd1)
    print("Installed Python versions:\n", s1.read().decode('utf-8', errors='ignore'))
    
    cmd2 = 'py -c "import torch; print(\'PyTorch:\', torch.__version__, \'CUDA available:\', torch.cuda.is_available(), \'Device:\', torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'None\')"'
    _, s2, e2 = c.exec_command(cmd2)
    print("PyTorch Check:\n", s2.read().decode('utf-8', errors='ignore'))
    print("PyTorch Error:\n", e2.read().decode('utf-8', errors='ignore'))
    
    c.close()
except Exception as e:
    print("Error:", e)
