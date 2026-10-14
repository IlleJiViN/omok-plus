<div align="center">
  <h1>🚀 fcolab</h1>
  <p><b>An elegant, context-manager-based Python wrapper for Headless Google Colab Execution</b></p>

  [![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Status: Experimental](https://img.shields.io/badge/Status-Experimental-red.svg)]()
  
  <p>Turn your local python script into a massive GPU beast using Google Colab's infrastructure, entirely in the background.</p>
</div>

---

## ⚡ Why fcolab?
Executing tasks on remote Google Colab GPUs manually via CLI can be tedious and dangerous. If your local script crashes, your remote Colab session might stay alive, leading to potential account bans or unexpected billing.

`fcolab` solves this by introducing a robust **Context Manager (`with` block)**. It guarantees that the remote GPU instance is cleanly terminated the moment your job finishes—or crashes. 

## 📦 Installation
Currently available via GitHub (Do not search on PyPI yet!):
```bash
pip install git+https://github.com/IlleJiViN/fcolab.git
```
*(Make sure to authenticate your Colab account via `colab login` in your terminal before first use.)*

## 💻 Quickstart

```python
from fcolab import FColab

# Start a Headless GPU worker (T4, V100, A100 supported)
with FColab(session_name="heavy_ai_worker", gpu="T4") as colab:
    
    print("Uploading massive datasets...")
    colab.upload("local_dataset.zip", "/content/dataset.zip")
    
    print("Training model on remote GPU...")
    colab.exec_script("train_model.py")
    
    print("Downloading weights...")
    colab.download("/content/model_weights.pt", "./model_weights.pt")

# 🛑 Session automatically destroyed! Safe from the Banhammer.
```

## ⚠️ DISCLAIMER & TERMS OF Service (철퇴 주의)
> **WARNING**: `fcolab` is an experimental tool built for research and extreme efficiency.
> 
> Heavy, automated, and continuous headless execution on Google Colab's Free Tier may violate Google's Terms of Service regarding resource abuse. Continuous mining, non-interactive deep learning farms, or proxying through Colab can and will result in a **Google Account Ban (철퇴)**.
> 
> **Use responsibly. We are not responsible if your Google account receives the banhammer.**

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for details.