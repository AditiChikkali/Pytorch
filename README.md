# Pytorch
Pytorch learning

## **1. PyTorch Core Features**
PyTorch is an open-source **machine learning framework** widely used for **deep learning and tensor computations**. Some of its core features include:

- **Tensor Computation:** Similar to NumPy arrays but with GPU acceleration.
- **Automatic Differentiation (Autograd):** Simplifies backpropagation by automatically computing gradients.
- **Dynamic Computation Graphs:** PyTorch builds computational graphs dynamically at runtime, making debugging easier.
- **GPU & TPU Acceleration:** Easily moves computations between CPU and GPU (`.to(device)`) for faster performance.
- **Deep Learning Model Support:** Includes modules for defining **neural networks (torch.nn)**, loss functions, and optimizers.
- **Integration with ML Libraries:** Supports **TorchVision (for images), TorchText (for NLP), and Hugging Face Transformers**.

---

## ** Where Are Tensors Used?**
Tensors are fundamental to **machine learning and deep learning**, acting as **multi-dimensional arrays** that store and manipulate data. They are used in:

- **Neural Networks:** Representing inputs, weights, and activations.
- **Image Processing:** Storing pixel values in multi-dimensional tensors (e.g., `(batch_size, channels, height, width)`).
- **Natural Language Processing (NLP):** Word embeddings and sentence representations.
- **Scientific Computing:** Simulating physical systems, quantum mechanics, or fluid dynamics.
- **Autonomous Systems & Robotics:** Processing sensor data.

---

## ** PyTorch Tensor Types**
Tensors in PyTorch are similar to NumPy arrays but can run on GPUs for speed. There are **different tensor types** based on their dimensions:

### **📌 Scalar (0D Tensor)**
A **scalar** is a single value (zero-dimensional tensor).
```python
import torch
scalar = torch.tensor(5)  
print(scalar.shape)  # Output: torch.Size([])
```

### **📌 Vector (1D Tensor)**
A **vector** is a one-dimensional tensor (array of numbers).
```python
vector = torch.tensor([2, 4, 6])  
print(vector.shape)  # Output: torch.Size([3])
```

### **📌 Matrix (2D Tensor)**
A **matrix** is a two-dimensional tensor (rows & columns).
```python
matrix = torch.tensor([[1, 2], [3, 4]])  
print(matrix.shape)  # Output: torch.Size([2, 2])
```

### **📌 3D Tensor (Tensors for RGB Images)**
A **3D tensor** extends matrices into an extra dimension.
```python
tensor3d = torch.rand(3, 3, 3)  # (3,3,3) shape
print(tensor3d.shape)  # Output: torch.Size([3, 3, 3])
```

### **📌 4D Tensor (Batch of Images)**
A **4D tensor** is commonly used in deep learning for images.
```python
tensor4d = torch.rand(32, 3, 64, 64)  # (batch, channels, height, width)
print(tensor4d.shape)  # Output: torch.Size([32, 3, 64, 64])
```

### **📌 5D Tensor (Videos in ML)**
A **5D tensor** is often used for **video data** in deep learning.
```python
tensor5d = torch.rand(10, 3, 16, 64, 64)  # (batch, channels, frames, height, width)
print(tensor5d.shape)  # Output: torch.Size([10, 3, 16, 64, 64])
```

---

## **2. What is Autograd?**
**Autograd (Automatic Differentiation)** in PyTorch automatically calculates **gradients** (derivatives) for tensor operations, which is essential for training deep learning models using **gradient descent**.

---

## ** Why Autograd?**
In deep learning, models learn by **adjusting weights using backpropagation**. PyTorch’s **autograd** automates this by:

✅ **Tracking operations on tensors with `requires_grad=True`**  
✅ **Computing gradients efficiently using `.backward()`**  
✅ **Reducing manual differentiation errors**  

Without autograd, computing derivatives manually would be inefficient.

---

## ** Example of Autograd in PyTorch**
### **📌 Step 1: Create a Tensor with `requires_grad=True`**
```python
import torch
# Create a tensor with autograd enabled
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2  # y = x^2

print(y)  # Output: tensor(9., grad_fn=<PowBackward0>)
```

Since `y` is derived from `x`, PyTorch tracks its computation graph.

---

### **📌 Step 2: Compute the Gradient**
To compute **dy/dx**, call `.backward()`:
```python
y.backward()  # Compute gradient of y with respect to x
print(x.grad)  # Output: tensor(6.)
```
Mathematically, **dy/dx = 2x = 2(3) = 6**.

---

### **📌 Example: Autograd in Neural Networks**
In a real-world deep learning model, autograd is used to update weights:
```python
import torch.nn as nn

# Define a simple neural network layer
linear = nn.Linear(2, 1)  # 2 inputs, 1 output
x = torch.tensor([1.0, 2.0])  # Input tensor
y = linear(x)  # Forward pass

# Compute gradient
y.backward(torch.ones_like(y))  # Backpropagation

# Get gradients of weights
print(linear.weight.grad)  # Output: tensor([[1., 2.]])
```

---




