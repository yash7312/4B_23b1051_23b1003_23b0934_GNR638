# Custom Deep Learning Framework (C++ Backend + Python Frontend)

This project implements a **custom deep learning framework from scratch** with a C++ computational backend and Python 3.12 bindings. The framework supports automatic differentiation, core neural network operations, and convolutional neural networks (CNNs) for **multiclass image classification** — without using any existing deep learning, numerical, or autograd libraries.

All models, training, evaluation, and metrics are built **exclusively using this framework**.

---

## Problem Statement

Design and implement a custom deep learning framework from first principles. The framework must support:

- Tensor abstractions with gradient tracking  
- Automatic differentiation and backpropagation  
- Core neural network operations  
- Convolutional layers, activations, pooling, and fully connected layers  

Using this framework, build and train a **CNN for multiclass classification** on the provided dataset.  
The computational backend must be implemented in **C++** for performance and exposed to **Python 3.12** via bindings (e.g., pybind11, ctypes, or cffi).

No third-party deep learning, numerical, or automatic differentiation libraries are permitted.

---

## Project Goals

By the end of this project, we deliver:

### 1 A Custom Deep Learning Framework

Implemented from scratch with:

- `Tensor` abstraction with:
  - Data, shape, gradients
  - Autograd graph and backpropagation

- Core operations:
  - Elementwise ops (add, mul, etc.)
  - MatMul
  - Conv2D
  - ReLU / activation
  - MaxPool
  - Flatten
  - Softmax

- Layers:
  - Conv2D
  - Activation
  - Pooling
  - Fully Connected (Linear / Dense)

- Loss:
  - Cross-Entropy (multiclass)

- Optimizer:
  - SGD (with optional momentum)

- Python bindings for training & evaluation

---

### 2 CNN Model

Built using only this framework with:

- ≥ 1 Convolution layer  
- ≥ 1 Activation layer  
- ≥ 1 Pooling layer  
- ≥ 1 Fully Connected layer  

Trained for multiclass image classification on the provided dataset only.

---

### 3 Built-in Metrics & Instrumentation

The framework exposes internal APIs for **trustable and reproducible analysis**:

✔️ Total number of trainable parameters  
✔️ MACs per forward pass  
✔️ FLOPs per forward pass  
✔️ Training & validation metrics across epochs:
- Loss
- Accuracy

✔️ Additional indicators:
- Per-layer parameter count
- Per-layer MACs / FLOPs
- Inference time per batch
- Gradient norms (optional)

✔️ Efficiency analysis:
- Memory consumption (parameters + activations)
- Model complexity vs performance trade-offs

Example API calls:
```cpp
model.num_params();
model.macs();
model.flops();
model.memory_bytes();

---

### 4 Training & Evaluation Pipeline

- Images are loaded using **OpenCV only**
- The entire training loop runs fully on the **custom framework**
- No external DL / numerical / autograd libraries are used

During training and validation, the framework logs:

- Epoch-wise **training loss**
- Epoch-wise **validation loss**
- **Training accuracy**
- **Validation accuracy**
- Total number of parameters
- MACs and FLOPs per forward pass
- Memory usage (parameters + activations)

Example metrics printed per epoch:
- Loss (train / val)
- Accuracy (train / val)
- Params, MACs, FLOPs
- Memory (MB)

---

### 5 Analysis & Report

The final report includes:

- Architecture description of the CNN
- Training and validation curves
- Parameter, MAC, and FLOP analysis
- Memory and efficiency analysis
- Discussion of one **failed design decision**, including:
  - What was attempted
  - Why it did not work
  - What was learned from the failure

- Model complexity vs performance trade-offs
- C++ backend vs Python frontend performance discussion

---

## Extensible Framework Design

The framework is modular and designed to be extended in the future with:

- Clean separation of:
  - Tensor / Autograd
  - Ops
  - Layers
  - Optimizers
  - Python bindings

This allows easy addition of new layers, optimizers, and backends.

---

## Future Roadmap (Planned Extensions)

To evolve this framework from basic to more powerful:

### Layers & Models
- Batch Normalization
- Dropout
- Residual / Skip Connections
- Deeper CNN Blocks

### Optimizers
- Adam
- RMSProp

### Performance
- im2col optimization
- Multi-threading (OpenMP)
- CUDA backend

### Autograd
- More efficient graph execution
- Gradient checkpointing

### Usability
- Model saving / loading
- ONNX export
- CLI training tools

### Monitoring
- Layer-wise profiling
- Memory & time tracing

---

## 🏁 Final Deliverables

✔️ Custom deep learning framework (C++ backend + Python frontend)  
✔️ Trained CNN model using only the custom framework  
✔️ Python training and evaluation scripts  
✔️ Built-in metric APIs (Params, MACs, FLOPs, Memory, Accuracy)  
✔️ Detailed report with analysis, efficiency study, and reflection  
