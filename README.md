# Phygrid CUDA PyTorch Image

[![Docker Hub](https://img.shields.io/docker/pulls/phygrid/cuda-pytorch.svg)](https://hub.docker.com/r/phygrid/cuda-pytorch)
[![Docker Image Version](https://img.shields.io/docker/v/phygrid/cuda-pytorch?sort=semver)](https://hub.docker.com/r/phygrid/cuda-pytorch/tags)
[![Build Status](https://github.com/phygrid/cuda-pytorch/workflows/Build%20and%20Deploy%20Docker%20Image/badge.svg)](https://github.com/phygrid/cuda-pytorch/actions)
[![License](https://img.shields.io/github/license/phygrid/cuda-pytorch)](LICENSE)

A multi-architecture Docker image optimized for PyTorch deep learning inference with GPU acceleration, supporting both Intel/AMD x64 systems and ARM64 NVIDIA Jetson devices. Built on the Phygrid CUDA base image with optimizations for NVIDIA Blackwell and earlier architectures.

## 🚀 Quick Start

```bash
# Pull the latest image
docker pull phygrid/cuda-pytorch:latest

# Use as base image in your Dockerfile
FROM phygrid/cuda-pytorch:1.0.0
```

## 📋 What's Included

### Base Layer
Built on `phygrid/cuda-base:latest` which includes:
- Python 3.11 with optimized pip, setuptools, wheel
- FastAPI, Uvicorn, Pydantic for web services
- Common system dependencies and security features

### PyTorch Ecosystem
- **PyTorch**: Version 2.8.0 with CUDA 12.8 (AMD64), PyTorch 2.5.0 with CUDA (ARM64 Jetson JetPack 6.1)
- **Transformers**: Hugging Face ecosystem (v4.36.2) with accelerate
- **Model Optimization**: bitsandbytes (AMD64), optimum for efficient inference
- **Hugging Face Hub**: Model management and safetensors support

### Computer Vision & Audio
- **Vision**: OpenCV, torchvision for image processing
- **Audio**: librosa, soundfile, torchaudio for audio processing
- **Scientific**: scipy, scikit-learn, matplotlib, seaborn

### GPU Optimizations
- **Memory Management**: Optimized CUDA allocation settings for both desktop and Jetson
- **Architecture Support**: Desktop GPUs (8.0-9.0), Jetson devices (5.3-8.7)
- **Environment Tuning**: Pre-configured for both Intel/AMD and ARM64 performance
- **Mixed Precision**: Automatic mixed precision support for efficient inference

## 🐳 Docker Hub

**Repository**: [phygrid/cuda-pytorch](https://hub.docker.com/r/phygrid/cuda-pytorch)

### Available Tags
- `latest` - Latest stable release
- `1.0.0`, `1.0.1`, etc. - Specific semantic versions
- Multi-architecture support: `linux/amd64`, `linux/arm64`

## 📦 Usage Examples

### As Base Image
```dockerfile
FROM phygrid/cuda-pytorch:1.0.0

# Copy your models and code
COPY models/ /app/pytorch_models/
COPY src/ /app/src/

# Install additional dependencies
RUN pip install -r requirements.txt

# Override default command
CMD ["python", "inference_server.py"]
```

### Development Environment
```bash
# Run interactive container with GPU support
docker run -it --rm \
  --gpus all \
  -v $(pwd):/app/workspace \
  -v $(pwd)/models:/app/pytorch_models \
  -v ~/.cache/huggingface:/app/cache/huggingface \
  -p 8000:8000 \
  phygrid/cuda-pytorch:latest \
  bash
```

### Production Deployment
```bash
# Intel/AMD systems with GPU support
docker run -d \
  --name pytorch-inference \
  --gpus all \
  -p 8000:8000 \
  -v /data/models:/app/pytorch_models \
  -v /data/cache:/app/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
  phygrid/cuda-pytorch:latest

# NVIDIA Jetson devices  
docker run -d \
  --name pytorch-inference \
  --runtime nvidia \
  --gpus all \
  -p 8000:8000 \
  -v /data/models:/app/pytorch_models \
  -v /data/cache:/app/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  phygrid/cuda-pytorch:latest
```

### Hugging Face Model Inference
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model with GPU support
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/app/cache/transformers")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/app/cache/transformers")

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Inference example
inputs = tokenizer.encode("Hello, how are you?", return_tensors="pt").to(device)
outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Custom Training/Fine-tuning
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Configure for RTX 5090
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Your training code here
model = nn.Sequential(...)
optimizer = torch.optim.AdamW(model.parameters())

# Enable mixed precision training
scaler = torch.cuda.amp.GradScaler()
```

## 🏗️ Building from Source

```bash
# Clone repository
git clone https://github.com/phygrid/cuda-pytorch.git
cd cuda-pytorch

# Build image
docker build -t phygrid/cuda-pytorch:custom .

# Build for multiple architectures
docker buildx build --platform linux/amd64,linux/arm64 -t phygrid/cuda-pytorch:custom .
```

## 🔄 Versioning

This project uses automated semantic versioning:

- **Automatic**: Patch versions increment on main branch changes
- **Manual**: Edit `VERSION` file for major/minor bumps
- **Tags**: Git tags created automatically (e.g., `v1.0.0`)

## 🧪 Health Check

The image includes a comprehensive health check:

```bash
# Test PyTorch setup
docker run --rm phygrid/cuda-pytorch:latest python /app/pytorch_test.py

# Expected output (with GPU):
# PyTorch version: 2.8.0
# Transformers version: 4.36.2
# ✅ CUDA available: NVIDIA GPU detected
#    CUDA version: 12.8 (AMD64) / 12.6 (ARM64 Jetson)
#    GPU memory: 24.0 GB (varies by device)
# 🎯 Detected NVIDIA Jetson device (ARM64 Jetson only)
# ✅ GPU tensor operations with mixed precision: OK
# PyTorch setup: OK
# PyTorch setup: OK
```

## ⚙️ Configuration

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:128` | CUDA memory allocation |
| `CUDA_LAUNCH_BLOCKING` | `0` | Enable for debugging |
| `TORCH_CUDA_ARCH_LIST` | `8.0;8.6;8.9;9.0` (AMD64) / `5.3;6.2;7.2;8.7` (ARM64) | Supported compute capabilities |
| `TRANSFORMERS_CACHE` | `/app/cache/transformers` | Hugging Face cache |
| `HF_HOME` | `/app/cache/huggingface` | HF Hub cache |
| `TOKENIZERS_PARALLELISM` | `true` | Enable tokenizer parallelism |

### Volume Mounts
| Path | Purpose |
|------|---------|
| `/app/pytorch_models` | PyTorch model storage |
| `/app/cache/transformers` | Hugging Face transformers cache |
| `/app/cache/huggingface` | Hugging Face Hub cache |
| `/app/cache/torch` | PyTorch cache |
| `/app/data` | Input/output data |
| `/app/logs` | Application logs |

## 🔧 Performance Tuning

### GPU-Specific Optimizations
```bash
# High-end desktop GPUs (NVIDIA Blackwell and earlier)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# NVIDIA Jetson devices (Orin, Xavier, Nano)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

# Enable TensorFloat-32 for faster training (desktop only)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
```

### Memory Management
```python
import torch

# Clear cache periodically
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
model.gradient_checkpointing_enable()

# Enable mixed precision
with torch.cuda.amp.autocast():
    outputs = model(inputs)
```

### Model Loading Optimization
```python
# Use memory mapping for large models
model = AutoModelForCausalLM.from_pretrained(
    "large-model",
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Clone and setup
git clone https://github.com/phygrid/cuda-pytorch.git
cd cuda-pytorch

# Test build locally
docker build -t phygrid/cuda-pytorch:test .
docker run --rm --gpus all phygrid/cuda-pytorch:test python /app/pytorch_test.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏷️ Labels

The image includes standard OCI labels:

```dockerfile
LABEL org.opencontainers.image.title="Phygrid CUDA PyTorch"
LABEL org.opencontainers.image.description="PyTorch base image for deep learning inference with GPU support"
LABEL org.opencontainers.image.vendor="Phygrid"
LABEL inference.engine="pytorch"
LABEL inference.runtime="pytorch-2.8.0"
LABEL cuda.version="12.8"
```

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/phygrid/cuda-pytorch/issues)
- **Discussions**: [GitHub Discussions](https://github.com/phygrid/cuda-pytorch/discussions)
- **Docker Hub**: [phygrid/cuda-pytorch](https://hub.docker.com/r/phygrid/cuda-pytorch)

## 📈 Metrics

- **Image size**: ~4.5GB compressed (AMD64), ~3.8GB (ARM64)
- **Build time**: ~15-25 minutes (with cache)
- **Architectures**: AMD64 (Intel/AMD), ARM64 (NVIDIA Jetson)
- **PyTorch version**: 2.8.0 with CUDA support
- **CUDA support**: 12.8 (ARM64 Jetson support)
- **GPU support**: NVIDIA Blackwell and earlier architectures
- **Jetson support**: Orin (8.7), Xavier (7.2), Nano (5.3)
- **Base image**: phygrid/cuda-base:latest

## 🎯 Use Cases

- **Large Language Models**: Run and fine-tune LLMs with Hugging Face
- **Computer Vision**: Image classification, object detection, segmentation
- **Audio Processing**: Speech recognition, audio generation
- **Research**: Rapid prototyping with pre-installed ML stack
- **Production**: High-performance inference services