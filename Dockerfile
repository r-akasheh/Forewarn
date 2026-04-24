FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libgl1-mesa-glx \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/bin:$PATH

# Ensure CUDA libraries are accessible
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /workspace

# Copy code into container
COPY . /workspace

# Create conda environment from environment-gemma4.yaml
RUN conda config --set auto_update_conda false && \
    conda config --set always_softlink false && \
    conda config --set auto_update_conda false && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f environment-gemma4.yaml && conda clean -afy

# ── CUDA / cuDNN library fixes ────────────────────────────────────────────────
# pip-installed nvidia-* packages land under site-packages/nvidia/*/lib but are
# NOT on LD_LIBRARY_PATH by default.  We (1) expose all of them, and (2) create
# the unversioned libnvrtc.so symlink that libcudnn_ops_infer.so.8 expects.
ENV NVIDIA_SITE_PKGS=/opt/conda/envs/dreamer/lib/python3.10/site-packages/nvidia

ENV LD_LIBRARY_PATH=\
${NVIDIA_SITE_PKGS}/cuda_nvrtc/lib:\
${NVIDIA_SITE_PKGS}/cudnn/lib:\
${NVIDIA_SITE_PKGS}/cublas/lib:\
${NVIDIA_SITE_PKGS}/cuda_runtime/lib:\
${NVIDIA_SITE_PKGS}/cufft/lib:\
${NVIDIA_SITE_PKGS}/curand/lib:\
${NVIDIA_SITE_PKGS}/cusolver/lib:\
${NVIDIA_SITE_PKGS}/cusparse/lib:\
${NVIDIA_SITE_PKGS}/nccl/lib:\
/usr/local/cuda/lib64:\
/usr/local/cuda/extras/CUPTI/lib64:\
$LD_LIBRARY_PATH

# Create unversioned symlinks so cuDNN 8 (CUDA 11-era) can find libnvrtc.so
RUN NVRTC_LIB=${NVIDIA_SITE_PKGS}/cuda_nvrtc/lib && \
    # Prefer .so.11.2 for cuDNN 8 compatibility; fall back to .so.12
    if [ -f "${NVRTC_LIB}/libnvrtc.so.11.2" ]; then \
        ln -sf ${NVRTC_LIB}/libnvrtc.so.11.2 ${NVRTC_LIB}/libnvrtc.so; \
    elif [ -f "${NVRTC_LIB}/libnvrtc.so.12" ]; then \
        ln -sf ${NVRTC_LIB}/libnvrtc.so.12 ${NVRTC_LIB}/libnvrtc.so; \
    fi && \
    # Register all nvidia lib dirs with the dynamic linker
    find ${NVIDIA_SITE_PKGS} -name "lib" -type d \
        | xargs -I{} sh -c 'echo {} >> /etc/ld.so.conf.d/nvidia-pip-pkgs.conf' && \
    ldconfig
# ─────────────────────────────────────────────────────────────────────────────

# Activate dreamer environment for all subsequent commands
SHELL ["conda", "run", "-n", "dreamer", "/bin/bash", "-c"]

# Reset shell to normal bash temporarily
SHELL ["/bin/bash", "-c"]

# Skip pip install -r requirements.txt since environment-gemma4.yaml already installed all packages
# The torch+cu118 version doesn't exist on PyPI anyway (conda handles CUDA variants)

# Re-activate conda shell for remaining commands
SHELL ["conda", "run", "-n", "dreamer", "/bin/bash", "-c"]

# Install dreamer world model
WORKDIR /workspace/model_based_irl_torch
RUN pip install -e .

# Skip llama-recipes install - will be installed via volume mount with proper SSH access
# WORKDIR /workspace/vlm/llama-recipes
# RUN pip install -e .

# Return to workspace root
WORKDIR /workspace

# Make sure conda environment is activated on container start
ENV CONDA_DEFAULT_ENV=dreamer
CMD ["/bin/bash"]