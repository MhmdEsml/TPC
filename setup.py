from setuptools import setup, find_packages

setup(
    name="tpcflib",
    version="0.1.0",
    description="High-performance Two-Point Correlation Function computation for 3D images",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
    ],
    extras_require={
        "torch": ["torch>=1.9"],
        "jax": ["jax>=0.3", "jaxlib>=0.3"],
        "all": ["torch>=1.9", "jax>=0.3", "jaxlib>=0.3"],
    },
    python_requires=">=3.8",
)