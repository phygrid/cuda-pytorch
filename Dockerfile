# Phygrid - PyTorch Base Image
# Optimized for deep learning inference with GPU acceleration
# Supports both Intel (x64) and ARM architectures

FROM phygrid/cuda-base:latest

# Switch to root for package installation
USER root

# Set architecture-aware variables
ARG TARGETARCH
ARG TARGETPLATFORM

# Install PyTorch with architecture-specific optimizations
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        # Intel x64 - Install with CUDA 12.8 support for RTX 5090
        echo "Installing PyTorch for x64 with CUDA 12.8 support..."; \
        python3 -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
        # ARM64 - Install PyTorch with CUDA for Jetson using direct wheel URLs
        echo "Installing PyTorch with CUDA for ARM64 Jetson (JetPack 6.1)..."; \
        python3 -m pip install --no-cache-dir https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl; \
        python3 -m pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cpu; \
        python3 -m pip install --no-cache-dir torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    else \
        # Fallback to CPU version
        echo "Installing PyTorch fallback (CPU only)..."; \
        python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install Transformers ecosystem with compatible versions
RUN python3 -m pip install --no-cache-dir \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    tokenizers==0.15.0 \
    datasets==2.16.1 \
    evaluate==0.4.1

# Install model optimization packages (skip bitsandbytes on ARM64 due to compatibility issues)
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        python3 -m pip install --no-cache-dir bitsandbytes==0.41.3; \
    fi && \
    python3 -m pip install --no-cache-dir optimum==1.16.1

# Install HuggingFace Hub and utilities
RUN python3 -m pip install --no-cache-dir \
    huggingface-hub==0.20.1 \
    safetensors==0.4.1

# Install computer vision and audio processing
RUN python3 -m pip install --no-cache-dir \
    opencv-python-headless==4.8.1.78 \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    scipy==1.11.4

# Install additional ML utilities
RUN python3 -m pip install --no-cache-dir \
    scikit-learn==1.3.2 \
    matplotlib==3.8.2 \
    seaborn==0.13.0

# Set PyTorch-specific environment variables
# For RTX 5090 (amd64) and Jetson (arm64) optimizations
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV CUDA_LAUNCH_BLOCKING=0
ENV TRANSFORMERS_CACHE=/app/cache/transformers
ENV HF_HOME=/app/cache/huggingface
ENV TOKENIZERS_PARALLELISM=true

# Set architecture-specific CUDA compute capabilities
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        # RTX 5090 and high-end desktop GPUs
        echo 'export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"' >> /etc/environment; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
        # Jetson devices (Orin: 8.7, Xavier: 7.2, Nano: 5.3)
        echo 'export TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2;8.7"' >> /etc/environment; \
    fi

# Create PyTorch-specific directories
RUN mkdir -p \
    /app/pytorch_models \
    /app/cache/transformers \
    /app/cache/huggingface \
    /app/cache/torch \
    && chown -R appuser:appuser /app/cache /app/pytorch_models

# Create PyTorch runtime test script
COPY --chown=appuser:appuser <<EOF /app/pytorch_test.py
#!/usr/bin/env python3
import torch
import transformers

def test_pytorch():
    try:
        print("PyTorch version:", torch.__version__)
        print("Transformers version:", transformers.__version__)
        
        # Test CUDA availability
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {device_name}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Test Jetson-specific features
            if "Tegra" in device_name or "Orin" in device_name or "Xavier" in device_name:
                print("🎯 Detected NVIDIA Jetson device")
                print(f"   GPU compute capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
        else:
            print("ℹ️  CUDA not available, using CPU")
            
        # Test basic tensor operation
        x = torch.randn(2, 3)
        if torch.cuda.is_available():
            x = x.cuda()
            # Test mixed precision on GPU
            with torch.cuda.amp.autocast():
                y = x * 2
            print("✅ GPU tensor operations with mixed precision: OK")
        else:
            y = x * 2
            print("✅ CPU tensor operations: OK")
        
        print("PyTorch setup: OK")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = test_pytorch()
    sys.exit(0 if success else 1)
EOF

RUN chmod +x /app/pytorch_test.py

# Switch back to non-root user
USER appuser

# Health check using PyTorch
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD python /app/pytorch_test.py

# Default command
CMD ["python", "/app/pytorch_test.py"]

# Labels
LABEL maintainer="Phygrid"
LABEL version="v1.0.8"
LABEL description="PyTorch base image for deep learning inference with GPU support"
LABEL inference.engine="pytorch"
LABEL inference.runtime="pytorch-2.8.0"
LABEL cuda.version="12.8"
