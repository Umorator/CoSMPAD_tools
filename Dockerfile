FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.cache/torch
ENV XDG_CACHE_HOME=/app/.cache

# ------------------------------
# System dependencies
# ------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------
# Install pip, setuptools, wheel properly
# ------------------------------
RUN python3 -m ensurepip --upgrade
RUN pip install --no-cache-dir --upgrade pip setuptools==69.5.1 wheel

# ------------------------------
# Install propy3
# ------------------------------
RUN pip install --no-cache-dir propy3

# ------------------------------
# Copy requirements first for better caching
# ------------------------------
COPY requirements.txt /app/
COPY . /app/

# ------------------------------
# CPU-only PyTorch
# ------------------------------
RUN pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# ------------------------------
# Install remaining dependencies including fair-esm
# ------------------------------
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fair-esm
RUN pip install --no-cache-dir .

# ------------------------------
# Pre-download ESM model during build using heredoc
# ------------------------------
RUN mkdir -p /app/.cache/torch/hub/checkpoints/ && \
    mkdir -p /app/.cache/esm && \
    python <<EOF
import os
import torch
import esm

print("=" * 50)
print("Downloading ESM-2 model (650M parameters)...")
print("=" * 50)

os.environ['TORCH_HOME'] = '/app/.cache/torch'
os.environ['XDG_CACHE_HOME'] = '/app/.cache'

# Download the model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

# Check if model was cached
model_path = '/app/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt'
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path) / (1024**3)
    print(f"✓ Model successfully cached at: {model_path}")
    print(f"✓ Model size: {file_size:.2f} GB")
else:
    print("⚠ Model not found in expected location, but should be cached")

# Verify model loading
print("\nVerifying model loading...")
device = torch.device('cpu')
model = model.to(device)
model.eval()
print("✓ Model verified and ready to use")
print("=" * 50)
EOF

# ------------------------------
# Verify model is cached
# ------------------------------
RUN ls -la /app/.cache/torch/hub/checkpoints/ || true

# ------------------------------
# Set final environment variables
# ------------------------------
ENV PYTHONPATH=/app:$PYTHONPATH

CMD ["python"]