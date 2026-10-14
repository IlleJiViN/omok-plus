import os
import subprocess

class FColab:
    """
    A Python wrapper for google-colab-cli.
    Allows for easy headless execution of Python scripts on Google Colab GPUs.
    
    Example:
        with FColab("my_session", gpu="T4") as colab:
            colab.upload("local.py", "/content/remote.py")
            colab.exec_script("remote.py")
            colab.download("/content/output.zip", "output.zip")
    """
    def __init__(self, session_name="default", gpu="T4"):
        self.session_name = session_name
        self.gpu = gpu

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _run_colab_cmd(self, args, error_msg):
        print(f"🚀 [FColab] Running: {' '.join(args)}")
        try:
            subprocess.run(args, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"{error_msg} : {e}")

    def start(self):
        print(f"🔥 [FColab] Starting session '{self.session_name}' with GPU {self.gpu}...")
        self._run_colab_cmd(["colab", "new", "-s", self.session_name, "--gpu", self.gpu], "Failed to start session")

    def upload(self, local_path, remote_path):
        print(f"📤 [FColab] Uploading {local_path} -> {remote_path}")
        self._run_colab_cmd(["colab", "upload", local_path, remote_path, "-s", self.session_name], "Failed to upload file")

    def exec_script(self, local_script_path):
        """
        Uploads and executes a local script on the remote Colab instance.
        If you want to execute a script that is already on Colab, use execute_remote() instead.
        """
        print(f"⚙️ [FColab] Executing local script {local_script_path} on Colab...")
        self._run_colab_cmd(["colab", "exec", "-s", self.session_name, "-f", local_script_path], "Failed to execute script")

    def execute_remote(self, command):
        """
        Executes an arbitrary shell command on the Colab instance.
        """
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(f"import os\nos.system('{command}')")
            tmp_path = f.name
            
        try:
            print(f"⚙️ [FColab] Executing remote command: {command}")
            self._run_colab_cmd(["colab", "exec", "-s", self.session_name, "-f", tmp_path], "Failed to execute remote command")
        finally:
            os.remove(tmp_path)

    def download(self, remote_path, local_path):
        print(f"📥 [FColab] Downloading {remote_path} -> {local_path}")
        self._run_colab_cmd(["colab", "download", remote_path, local_path, "-s", self.session_name], "Failed to download file")

    def stop(self):
        print(f"🛑 [FColab] Stopping session '{self.session_name}' to prevent billing...")
        self._run_colab_cmd(["colab", "stop", "-s", self.session_name], "Failed to stop session")