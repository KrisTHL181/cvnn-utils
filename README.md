# 🌊 cvnn-utils - A *FULL-COMPLEX* CVNN Tool Pack

> **ℂ is not ℝ².**
> *A lightweight, mathematically rigorous toolkit for building truly complex-valued neural networks in PyTorch.*

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange.svg)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://python.org)
[![Code style](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)

---

## 🚨 The Problem: Most "Complex" NNs Are Wrong

Most so-called "complex-valued neural networks" (CVNNs) in deep learning **are not truly complex** — they treat ℂ as ℝ² by splitting real/imaginary parts and processing them separately. This approach:

- ❌ **Breaks complex linearity**: `f(az) ≠ a·f(z)` for `a ∈ ℂ`
- ❌ **Destroys phase equivariance**: Rotating input phase doesn't rotate output phase
- ❌ **Violates Wirtinger calculus**: Invalid gradients in complex domain
- ❌ **Ignores algebraic structure**: Treats ℂ as just two real numbers

> **This is like building a quantum computer that treats qubits as classical bits.**
> *You're missing the entire point of using complex numbers.*

---

## ✅ The Solution: Respect ℂ as a Field

**cvnn-utils** provides *mathematically correct* implementations of neural network components that:

- ✅ **Preserve complex linearity**: `f(az) = a·f(z)`
- ✅ **Maintain phase equivariance**: Input phase rotation → Output phase rotation
- ✅ **Use proper complex differentiation**: Valid Wirtinger gradients
- ✅ **Respect ℂ algebraic structure**: No arbitrary ℝ² splitting

> **This is how complex-valued deep learning should be done.**

---

## 📦 Key Features

### 🔹 **Mathematically Sound Components**

- `ComplexStandardBatchNorm2d`: Proper complex batch norm (not covariance-based)
- `ComplexGlobalAvgPool2d`: The *only* mathematically valid complex pooling
- `ComplexModLeakyReLU`: Phase-invariant activation (|z| based)
- `ComplexConv2d` & `ComplexLinear`: Proper complex weight initialization

### 🔹 **Dangerous Operation Warnings**

- ⚠️ Blocks ill-defined operations like `ComplexAvgPool2d` by default
- 🚫 Explicit warnings when using `allow_inconsistencies=True`
- 💡 Educational messages explaining *why* certain operations are problematic

### 🔹 **Philosophy-Driven Design**

- **No `ComplexToReal` trap**: Forces users to make conscious design choices
- **Clear documentation**: Every class explains its mathematical properties
- **No false abstractions**: Only provides operations that respect ℂ structure

---

## 🚀 Quick Start

### Installation

```bash
pip install git+https://github.com/KrisTHL181/cvnn-utils.git
```

### Basic Usage

```python
import cvnn_utils as cvnn
import torch

# Create a truly complex network
model = torch.nn.Sequential(
    cvnn.ComplexConv2d(3, 64, 3, padding=1),
    cvnn.ComplexStandardBatchNorm2d(64),
    cvnn.ComplexModLeakyReLU(64),
    cvnn.ComplexConv2d(64, 64, 3, stride=2, padding=1),  # not pooling!
    torch.nn.Flatten()
)

# Get complex output
z = model(torch.randn(16, 3, 32, 32).to(torch.complex64))

# Now decide HOW to map to real logits (your design choice):
logits = z.real @ W_r.T + z.imag @ W_i.T + b  # Option 1
# OR
logits = classifier(z.abs())  # Option 2
# OR
logits = (z * w.conj()).real.sum(dim=1)  # Option 3
```

### Why This Matters: Phase Equivariance Test

```python
# Test phase rotation invariance
theta = 0.785  # π/4
phase = torch.exp(1j * theta)

z = torch.randn(1, 3, 32, 32, dtype=torch.complex64)
z_rotated = z * phase

output = model(z)
output_rotated = model(z_rotated)

# In a proper CVNN:
assert torch.allclose(output_rotated, output * phase, atol=1e-5)
```

---

## 🧠 Core Philosophy: ℂ is Not ℝ²

### The Critical Mistake

Most CVNN implementations commit this error:

```python
# WRONG: Treats complex as two real channels
real_out = F.conv2d(x.real, weight.real) - F.conv2d(x.imag, weight.imag)
imag_out = F.conv2d(x.real, weight.imag) + F.conv2d(x.imag, weight.real)
```

This **breaks complex linearity** because:

```
f(az) = f(a·(x+iy)) ≠ a·f(z) = a·(real_out + i·imag_out)
```

### The Correct Approach

A *true* complex operation satisfies:

```
f(az) = a·f(z) for all a ∈ ℂ
```

Which requires:

```python
# CORRECT: Proper complex convolution
output = F.conv2d(x, weight)  # PyTorch natively supports complex conv!
```

**cvnn-utils** ensures all operations respect this fundamental property.

---

## 📚 Supported Components

| Component                        | Status | Mathematical Properties                     |
| -------------------------------- | ------ | ------------------------------------------- |
| `ComplexConv2d`                | ✅     | ℂ-linear, phase equivariant                |
| `ComplexLinear`                | ✅     | ℂ-linear, phase equivariant                |
| `ComplexStandardBatchNorm2d`   | ✅     | ℂ-linear, phase equivariant                |
| `ComplexCovarianceBatchNorm2d` | ⚠️   | NOT ℂ-linear (for non-circular data only)  |
| `ComplexGlobalAvgPool2d`       | ✅     | Only valid complex pooling                  |
| `ComplexModLeakyReLU`          | ✅     | Phase invariant,                            |
| `ComplexDropout`               | ✅     | ℂ-linear variants                          |
| `ComplexAvgPool2d`             | ❌     | BLOCKED by default (mathematically invalid) |
| `ComplexAdaptiveAvgPool2d`     | ❌     | BLOCKED for output_size > 1                 |

---

## ❓ Why This Matters: Real-World Impact

### In Signal Processing Tasks:

- **Radar/Communication**: Preserves phase information critical for direction finding
- **MRI Reconstruction**: Maintains complex coil sensitivity relationships
- **Audio Processing**: Keeps phase coherence in STFT representations

### In Computer Vision:

- **Rotation-invariant features**: Phase equivariance enables better rotation handling
- **Frequency-domain learning**: Proper complex ops are essential for FFT-based networks
- **Polar representation**: |z| (magnitude) and arg(z) (phase) have distinct meanings

> **When you break complex structure, you lose these advantages.**

---

## 📖 Documentation

Each component includes **clear documentation of its mathematical properties**:

```python
class ComplexStandardBatchNorm2d(ComplexModule):
    """
    Standard Complex Batch Normalization.
  
    Preserves:
        - ✅ Complex linearity (f(az) = a f(z), a ∈ ℂ)
        - ✅ Phase equivariance (rotation-invariant up to scale)
        - ✅ C-differentiability (Wirtinger sense)
  
    Does NOT model:
        - ❌ Non-circularity (improperness): E[z^2] ≠ 0
  
    Use this as the default BN in most CVNN applications.
    """
```

---

## 🤝 Contributing

We welcome contributions that:

- Add mathematically sound complex operations
- Improve documentation with rigorous explanations
- Create tutorials demonstrating proper CVNN usage
- Develop tests verifying complex properties

**Please avoid**:

- Adding operations that treat ℂ as ℝ² without warning
- Creating abstractions that hide mathematical choices
- Implementing "convenience" layers that encourage bad practices

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---


## 🌟 Join the Movement

Stop pretending complex numbers are just two real numbers.
Start building neural networks that **respect the algebraic structure of ℂ**.

**cvnn-utils** is the first step toward *mathematically correct* complex-valued deep learning.

---

> ℂ is not ℝ².
> **Respect the field.**

[![GitHub stars](https://img.shields.io/github/stars/KrisTHL181/cvnn-utils?style=social)](https://github.com/KrisTHL181/cvnn-utils/stargazers)
*Star this repo if you believe complex-valued deep learning deserves mathematical rigor.*
