from setuptools import setup, find_packages
import os


def get_version():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "cvnn_utils", "__init__.py"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip("\"'")
    raise RuntimeError("Cannot find version")


setup(
    name="cvnn-utils",
    version=get_version(),
    description="A lightweight, principled toolkit for Complex-Valued Neural Networks in PyTorch.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="KrisTHL181",
    url="https://github.com/KrisTHL181/cvnn-utils",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    keywords="deep learning, complex-valued, neural network, cvnn, pytorch",
)
