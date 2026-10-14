from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fcolab",
    version="0.1.0",
    author="IlleJiViN",
    description="An elegant context-manager wrapper for google-colab-cli",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IlleJiViN/fcolab",
    packages=find_packages(),
    install_requires=[
        "google-colab-cli>=0.1.5"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)