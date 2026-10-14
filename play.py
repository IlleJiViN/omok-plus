"""
Omok++ Quick Launcher
Directly launch the interactive Omok++ desktop GUI from any working directory.
"""

import os
import sys

# Compute absolute path to project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Change current working directory to PROJECT_ROOT
# This guarantees that any relative file/module accesses work properly regardless of where it was launched
os.chdir(PROJECT_ROOT)

# Ensure project root is at the front of sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gui.omok_gui import launch

if __name__ == "__main__":
    print(f"Launching Omok++ AI Arena GUI from: {PROJECT_ROOT}")
    launch()
