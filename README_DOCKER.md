# Docker Setup Guide

## Quick Start

### Build the Docker Image

```bash
docker-compose build aircraft-control
```

This builds a complete environment with:
- JSBSim (compiled from source)
- All Python dependencies
- C++ build tools and Pybind11
- Headless graphics support (Xvfb)
- pyproj for accurate geodetic conversion

### Run Interactive Development Environment

```bash
docker-compose run --rm aircraft-control
```

This starts an interactive bash shell with access to:
- Full codebase (mounted at `/workspace`)
- Pre-built C++ extensions
- JSBSim Python bindings

### Run Training (Headless)

```bash
# Single-agent training
docker-compose up training

# Multi-agent training
docker-compose up multi-agent-training
```

### Run Dashboard

```bash
docker-compose up dashboard
```

Access at: http://localhost:8050

### Run TensorBoard

```bash
docker-compose up tensorboard
```

Access at: http://localhost:6006

---

## Environment Variables

### `HEADLESS`
- **Default**: `false`
- **Options**: `true`, `false`
- **Description**: Enable headless rendering (no display required)

Example:
```bash
docker-compose run -e HEADLESS=true aircraft-control python examples/train_headless.py
```

### `OMP_NUM_THREADS`
- **Default**: Not set
- **Description**: Limit OpenMP threads for parallel computation

---

## Usage Examples

### Train RL Agent (Headless, Fast)

```bash
docker-compose run -e HEADLESS=true training \
    python training/train_level3_ppo.py \
    --config configs/simulation/ultra_fast_training.yaml \
    --timesteps 1000000
```

### Run Multi-Agent Formation Flying

```bash
docker-compose run -e HEADLESS=true multi-agent-training \
    python training/train_multi_agent_rllib.py \
    --config configs/multi_agent/formation_4_aircraft.yaml
```

### Export Model for Deployment

```bash
docker-compose run aircraft-control \
    python deployment/export_onnx.py \
    --model models/level3_ppo.zip \
    --output models/level3_ppo.onnx
```

### Run HIL Validation

```bash
# Requires hardware connection (add device mapping)
docker-compose run --device=/dev/ttyACM0 aircraft-control \
    python validation/hil_test.py \
    --model models/level3_ppo_quantized.onnx \
    --episodes 50
```

---

## Development Workflow

### 1. Build Image (Once)

```bash
docker-compose build
```

### 2. Develop Inside Container

```bash
docker-compose run --rm aircraft-control bash
```

Inside container:
```bash
# Make code changes (files are mounted from host)
vim controllers/my_controller.py

# Rebuild C++ if needed
pip install -e .

# Run tests
pytest tests/

# Train agent
python training/train_level3_ppo.py
```

### 3. View Results

Logs and models are saved to mounted volumes:
- `./logs` → Training logs, TensorBoard
- `./models` → Saved models

Access from host machine after container exits.

---

## Headless Rendering

The Docker image includes Xvfb (X Virtual Framebuffer) for headless operation.

### Automatic Xvfb (via entrypoint)

Set `HEADLESS=true`:
```bash
docker-compose run -e HEADLESS=true aircraft-control python my_script.py
```

### Manual Xvfb

```bash
docker-compose run aircraft-control bash
# Inside container:
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
python training/train_with_3d_viz.py
```

### Save Screenshots/Videos

Configure in YAML:
```yaml
visualization:
  3d_viewer:
    headless: true
    save_screenshots: true
    screenshot_dir: "logs/screenshots"
```

Convert screenshots to video:
```bash
ffmpeg -framerate 30 -pattern_type glob -i 'logs/screenshots/*.png' \
    -c:v libx264 -pix_fmt yuv420p logs/videos/episode.mp4
```

---

## Resource Limits

### CPU Limits

Edit `docker-compose.yml`:
```yaml
services:
  training:
    deploy:
      resources:
        limits:
          cpus: '8'  # Limit to 8 cores
```

### Memory Limits

```yaml
services:
  training:
    deploy:
      resources:
        limits:
          memory: 16G  # Limit to 16GB RAM
```

### GPU Support (Optional)

For GPU-accelerated training:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Update `docker-compose.yml`:
```yaml
services:
  training:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

3. Run:
```bash
docker-compose up training
```

---

## Troubleshooting

### JSBSim Import Error

**Problem**: `ModuleNotFoundError: No module named 'jsbsim'`

**Solution**: Rebuild image to ensure JSBSim is compiled:
```bash
docker-compose build --no-cache
```

### Permission Issues

**Problem**: Files created inside container are owned by root

**Solution**: Run with user ID mapping:
```bash
docker-compose run --user $(id -u):$(id -g) aircraft-control
```

Or add to `docker-compose.yml`:
```yaml
services:
  aircraft-control:
    user: "${UID}:${GID}"
```

Then run:
```bash
export UID=$(id -u)
export GID=$(id -g)
docker-compose run aircraft-control
```

### Display/Rendering Issues

**Problem**: PyVista fails with "Could not open display"

**Solution**: Enable headless mode:
```bash
docker-compose run -e HEADLESS=true aircraft-control python script.py
```

Or verify Xvfb is running:
```bash
ps aux | grep Xvfb
```

---

## Multi-Stage Build Details

The Dockerfile uses multi-stage build for efficiency:

**Stage 1 (builder)**:
- Compiles JSBSim from source
- Builds Python bindings
- ~1.5GB build context

**Stage 2 (final)**:
- Copies only compiled artifacts
- Installs runtime dependencies
- Final image: ~800MB

This reduces final image size by ~50%.

---

## Alternative: Conda Environment (Without Docker)

If you prefer not to use Docker:

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Create environment:
```bash
conda env create -f environment.yml
conda activate aircraft-control
```

3. Build JSBSim manually:
```bash
git clone https://github.com/JSBSim-Team/jsbsim.git
cd jsbsim
mkdir build && cd build
cmake .. -DBUILD_PYTHON_MODULE=ON
make -j$(nproc)
sudo make install
```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

---

## Summary

**For training (fast, headless)**:
```bash
docker-compose up training
```

**For development (interactive)**:
```bash
docker-compose run --rm aircraft-control
```

**For visualization**:
```bash
docker-compose up dashboard
```

**For production deployment**: Export models and deploy to target hardware (Jetson, RPi, Teensy).
