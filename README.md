# TPCFLib: High-Performance Two-Point Correlation Function Computation

<p align="center">
  <img src="[[https://via.placeholder.com/800x200/2E86AB/FFFFFF?text=TPCFLib+%E2%80%93+Fast+Two-Point+Correlation+Functions](https://github.com/MhmdEsml/TPC/blob/main/0.png)](https://github.com/MhmdEsml/TPC/blob/main/0.png)" alt="TPCFLib Banner" width="800">
</p>

<p align="center">
  <a href="https://python.org/">
    <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python version">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  </a>
</p>

## 📖 Overview

**TPCFLib** is a high-performance Python library for computing the Two-Point Correlation Function (TPCF) $S_2(r)$ of 3D porous materials and binary microstructures. The library provides accelerated computation using both PyTorch and JAX backends, offering significant speedups over conventional implementations.

The two-point correlation function $S_2(r)$ is defined as:

$$
S_2(r) = \langle I(\mathbf{x}) I(\mathbf{x} + \mathbf{r}) \rangle
$$

where $I(\mathbf{x})$ is the indicator function of the phase of interest at position $\mathbf{x}$, and $\langle \cdot \rangle$ denotes ensemble averaging.

## 🚀 Features

- **Multi-Backend Support**: Choose between PyTorch and JAX for optimal performance
- **Batch Processing**: Efficient computation on multiple 3D volumes simultaneously
- **Porous Media Datasets**: Built-in access to standard porous media datasets
- **GPU/TPU Acceleration**: Automatic GPU?TPU utilization when available (using JAX for TPU)

## 📦 Installation

### Installation
```bash
pip install tpc
