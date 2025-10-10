# Multi-stage Dockerfile for Aircraft Control System
# Includes JSBSim, all Python dependencies, and optional headless support

FROM ubuntu:22.04 as builder

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    libexpat1-dev \
    && rm -rf /var/lib/apt/lists/*

# Build JSBSim from source
WORKDIR /tmp
RUN git clone https://github.com/JSBSim-Team/jsbsim.git && \
    cd jsbsim && \
    mkdir build && \
    cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_PYTHON_MODULE=ON \
        -DBUILD_SHARED_LIBS=ON && \
    make -j$(nproc) && \
    make install

# ============================================
# Final image
# ============================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libexpat1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install headless graphics support (for optional 3D visualization)
RUN apt-get update && apt-get install -y \
    xvfb \
    libgl1-mesa-glx \
    libglu1-mesa \
    libegl1-mesa \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy JSBSim from builder
COPY --from=builder /usr/local/lib/python3* /usr/local/lib/
COPY --from=builder /usr/local/lib/libJSBSim* /usr/local/lib/
COPY --from=builder /usr/local/include/JSBSim /usr/local/include/JSBSim
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages:$PYTHONPATH

# Create working directory
WORKDIR /workspace

# Copy requirements
COPY requirements.txt /workspace/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install pyproj for accurate geodetic conversion
RUN pip3 install --no-cache-dir pyproj

# Install additional RL/visualization packages
RUN pip3 install --no-cache-dir \
    stable-baselines3[extra] \
    ray[rllib] \
    tensorboard \
    wandb \
    optuna \
    pyvista \
    plotly \
    dash \
    dash-bootstrap-components

# Copy project files
COPY . /workspace/

# Build C++ extensions
RUN pip3 install --no-cache-dir -e .

# Set up X virtual framebuffer for headless operation
ENV DISPLAY=:99

# Create entrypoint script
RUN echo '#!/bin/bash\n\
# Start Xvfb in background if HEADLESS=true\n\
if [ "$HEADLESS" = "true" ]; then\n\
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\n\
    export DISPLAY=:99\n\
fi\n\
\n\
# Execute command\n\
exec "$@"\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]
