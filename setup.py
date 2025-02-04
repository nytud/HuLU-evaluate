from setuptools import find_packages, setup
import subprocess
import platform
import sys


def check_cuda_available() -> bool:
    try:
        result = subprocess.run(
            ["nvcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


install_requires = open("requirements.txt", "r").read().splitlines()


if platform.system() == "Windows" and not check_cuda_available():
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch==2.3.1+cpu",
                "-f",
                "https://download.pytorch.org/whl/torch_stable.html",
            ]
        )
    except subprocess.CalledProcessError:
        print("Failed to install torch with CPU-only version.")
else:
    install_requires.append("torch==2.4.0")

extras = {
    "dev": ["ruff>=0.5.0", "black", "isort"],
}

setup(
    name="hulu_evaluate",
    version="0.0.2",
    author="HUN-REN Research Center for Linguistics",
    author_email="varga.kristof@nytud.hun-ren.hu",
    description="Client library to train and evaluate models on the HuLu benchmark.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="machine-learning models natural-language-processing deep-learning evaluation benchmark",
    license="Apache",
    url="https://hulu.nytud.hu/",
    package_dir={"": "src"},
    packages=find_packages("src"),
    extras_require=extras,
    entry_points={
        "console_scripts": ["hulu-evaluate=cli.cli:cli"],
    },
    python_requires=">=3.8.0",
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
)
