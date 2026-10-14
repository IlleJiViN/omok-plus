"""
Setup configuration for the AlphaOmok Python package.
"""

from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

# Read README if available
readme = ""
readme_path = os.path.join(here, "README.md")
if os.path.exists(readme_path):
    with open(readme_path, encoding="utf-8") as f:
        readme = f.read()

setup(
    name="alphaomok",
    version="1.0.0",
    description="AlphaOmok: Production-grade Deep RL & PUCT MCTS SDK for Omok++",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="AlphaOmok Team",
    packages=find_packages(include=["alphaomok", "alphaomok.*"]),
    package_data={
        "alphaomok": [
            "models/weights/*.pt",
        ],
    },
    include_package_data=True,
    install_requires=[
        "torch",
        "numpy",
        "numba",
    ],
    entry_points={
        "console_scripts": [
            "alphaomok = alphaomok.cli:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Board Games",
    ],
)
