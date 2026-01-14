# Frequently Asked Questions (FAQ)

Common questions and troubleshooting for the Multi-Level Flight Control platform.

---

## Table of Contents

1. [General Questions](#general-questions)
2. [Installation & Setup](#installation--setup)
3. [Running Examples](#running-examples)
4. [RL Training](#rl-training)
5. [Performance & Optimization](#performance--optimization)
6. [Research & Development](#research--development)
7. [Hardware & Deployment](#hardware--deployment)

---

## General Questions

### What is this project?

This is a research platform for developing and comparing flight control algorithms across multiple levels of abstraction. It implements a 5-level cascaded control hierarchy (similar to ArduPilot/PX4) and supports both classical PID and reinforcement learning controllers.

**Key features**:
- Train RL agents at any control level (waypoint, HSA, attitude, rate, surface)
- Compare RL vs PID performance quantitatively
- 6-DOF physics simulation validated against JSBSim
- 132 comprehensive tests, ~24k lines of production-quality code

### Who is this for?

- **Researchers**: Study multi-level RL, sim-to-real transfer, hybrid control
- **Students**: Learn flight control theory and RL applications
- **Engineers**: Develop advanced flight controllers or autonomous systems
- **Hobbyists**: Build custom aircraft controllers for RC planes/drones

### How is this different from ArduPilot/PX4?

**Similarities**:
- 5-level cascaded control hierarchy
- Industry-standard architecture (rate inner loop, attitude outer loop)
- PID-based classical controllers

**Differences**:
- **Research focus**: Not production-ready, optimized for experimentation
- **RL integration**: Train learned controllers at any level
- **Python-first**: Easier to modify and experiment (C++ only for inner loop)
- **Simulation-native**: Designed for fast RL training
- **Multi-level learning**: Compare algorithms across abstraction levels

### Can I use this on a real aircraft?

**Not yet** - Phase 6 (Hardware Interface) is planned but not implemented. Currently simulation-only.

**Future**: Hardware integration is planned with Teensy 4.1 + MAVLink. See [Implementation Roadmap](design_docs/12_IMPLEMENTATION_ROADMAP.md).

### Is this safe for real flight?

**No** - This is research code, not safety-certified. Do not use on crewed aircraft.

If/when hardware support is added, extensive testing and safety systems (failsafes, geofencing, watchdogs) will be required.

---

## Installation & Setup

### Installation fails with "ModuleNotFoundError"

**Problem**: Missing dependencies after `pip install -r requirements.txt`

**Solutions**:
```bash
# 1. Ensure you're in the virtual environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install dependencies again
pip install -r requirements.txt

# 4. For RL features, install extras
pip install stable-baselines3[extra] sb3-contrib
```

### C++ build fails with "CMake not found"

**Problem**: CMake not installed or not in PATH

**Solutions**:
```bash
# Ubuntu/Debian
sudo apt-get install cmake build-essential

# macOS
brew install cmake

# Windows
# Download from https://cmake.org/download/
# Or use: choco install cmake
```

### C++ build fails with "pybind11 not found"

**Problem**: Pybind11 not installed

**Solution**:
```bash
pip install pybind11
```

### HDF5 errors on macOS

**Problem**: `ImportError: dlopen(...libhdf5.dylib): image not found`

**Solution**:
```bash
# Install HDF5 via Homebrew
brew install hdf5

# Set environment variable
export HDF5_DIR=/opt/homebrew/opt/hdf5  # or /usr/local/opt/hdf5

# Reinstall h5py from source
pip uninstall h5py
pip install h5py --no-binary=h5py
```

### Pygame window doesn't open

**Problem**: GUI examples don't display window

**Solutions**:
```bash
# 1. Ensure display is available (not running headless)
echo $DISPLAY  # Should show :0 or similar

# 2. On Linux, set video driver
export SDL_VIDEODRIVER=x11

# 3. Update Pygame
pip install --upgrade pygame

# 4. Install system dependencies (Ubuntu)
sudo apt-get install python3-pygame

# 5. Test Pygame installation
python -c "import pygame; pygame.init(); print('Pygame works!')"
```

### Import error: "cannot import name 'RecurrentPPO'"

**Problem**: Missing SB3-Contrib package

**Solution**:
```bash
pip install sb3-contrib
```

---

## Running Examples

### "Model not found" error in RL examples

**Problem**: `examples/02_rl_vs_pid_demo.py` can't find trained model

**Solution**:
The demo gracefully handles missing models and runs PID-only comparison. To add RL:

```bash
# Option 1: Train your own model (recommended)
cd learned_controllers
python train_rate.py  # Takes ~30 min on CPU

# Option 2: Download pretrained model (if available)
# wget https://github.com/.../rate_controller_best.zip
# mv rate_controller_best.zip learned_controllers/models/

# Option 3: Run PID-only comparison
python examples/02_rl_vs_pid_demo.py  # Works without RL model
```

### Simulation runs very slowly

**Problem**: Simulation takes too long

**Solutions**:
1. **Disable real-time plotting** - Comment out `plt.pause()` calls
2. **Reduce simulation duration** - Decrease `sim_time` parameter
3. **Increase timestep** - Use `dt=0.02` instead of `dt=0.01`
4. **Check CPU usage** - Should use ~100% of one core
5. **Close other applications** - Free up system resources

### GUI is laggy or stuttering

**Problem**: Pygame GUI not running smoothly

**Solutions**:
1. **Reduce visual quality**:
   ```python
   # In GUI code, reduce update frequency
   if frame % 2 == 0:  # Update every other frame
       update_display()
   ```

2. **Disable 3D rendering** - Comment out 3D aircraft visualization

3. **Use simpler aircraft model** - Reduce polygon count

4. **Check GPU acceleration** - Ensure Pygame is using hardware acceleration

### Tests fail with "fixture not found"

**Problem**: Pytest can't find fixtures

**Solution**:
```bash
# Ensure you're running from project root
cd /path/to/controls

# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Run tests
pytest tests/
```

---

## RL Training

### Training is very slow

**Problem**: RL training taking many hours

**Solutions**:
1. **Use GPU if available**:
   ```bash
   # Check if CUDA is available
   python -c "import torch; print(torch.cuda.is_available())"

   # Modify training script to use GPU
   device = "cuda" if torch.cuda.is_available() else "cpu"
   ```

2. **Reduce total timesteps**:
   ```python
   # In train_rate.py
   total_timesteps = 500_000  # Instead of 1_000_000
   ```

3. **Vectorize environments**:
   ```python
   from stable_baselines3.common.vec_env import SubprocVecEnv
   env = SubprocVecEnv([make_env() for _ in range(4)])  # 4 parallel envs
   ```

4. **Use faster simulation backend** - Ensure C++ components are built

### Training crashes with "CUDA out of memory"

**Problem**: GPU memory exhausted

**Solutions**:
```python
# 1. Use CPU instead
model = RecurrentPPO(..., device="cpu")

# 2. Reduce batch size
model = RecurrentPPO(..., batch_size=64)  # Instead of 128

# 3. Reduce network size
policy_kwargs = dict(net_arch=[64, 64])  # Instead of [256, 256]

# 4. Clear CUDA cache
import torch
torch.cuda.empty_cache()
```

### Model doesn't learn / reward stays flat

**Problem**: RL training not improving

**Debugging steps**:
1. **Check reward scale**:
   ```python
   # Print rewards during training
   print(f"Mean reward: {np.mean(episode_rewards)}")
   # Should be changing over time
   ```

2. **Verify environment**:
   ```python
   # Test environment manually
   env = RateControlEnv()
   obs = env.reset()
   for _ in range(100):
       action = env.action_space.sample()
       obs, reward, done, info = env.step(action)
       print(f"Reward: {reward}")
   ```

3. **Adjust learning rate**:
   ```python
   model = RecurrentPPO(..., learning_rate=1e-4)  # Try different values
   ```

4. **Simplify task**:
   ```python
   # Start with easier scenario
   difficulty = "easy"  # Instead of "medium" or "hard"
   ```

5. **Check observation/action spaces**:
   ```python
   # Ensure obs is normalized
   obs = (obs - obs_mean) / (obs_std + 1e-8)
   ```

### How do I resume interrupted training?

**Solution**:
```bash
# Training auto-saves checkpoints
# Resume from latest checkpoint
python learned_controllers/recover_training.py \
  --checkpoint learned_controllers/checkpoints/rate_controller_500000_steps.zip
```

Or manually:
```python
# Load existing model
model = RecurrentPPO.load("checkpoints/rate_controller_500000_steps.zip")

# Continue training
model.learn(total_timesteps=500_000, reset_num_timesteps=False)
```

---

## Performance & Optimization

### How fast should simulation run?

**Expected performance**:
- **Simplified 6-DOF**: ~1000 simulation steps/second (CPU)
- **RL training**: ~1000-5000 environment steps/second
- **GUI**: 30-60 FPS real-time

**Benchmarking**:
```bash
# Time a simulation
time python examples/01_hello_controls.py
# Should complete 5s simulation in <1s wall time
```

### Can I use multiple CPUs for training?

**Yes!** Use vectorized environments:

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

# Create 8 parallel environments
envs = SubprocVecEnv([make_env for _ in range(8)])

# Train with parallel rollouts
model = RecurrentPPO("MlpLstmPolicy", envs, ...)
model.learn(total_timesteps=1_000_000)
```

**Speedup**: ~4-8x on multi-core CPUs

### How do I profile performance?

**Python profiling**:
```bash
# Install profiler
pip install py-spy

# Profile example
py-spy record -o profile.svg -- python examples/01_hello_controls.py

# View profile
open profile.svg
```

**Detailed profiling**:
```python
import cProfile
import pstats

cProfile.run('run_simulation()', 'profile_stats')
p = pstats.Stats('profile_stats')
p.sort_stats('cumulative').print_stats(20)
```

---

## Research & Development

### Which control level should I use RL for?

**Depends on your research question:**

**Level 4 (Rate)**:
- **Pros**: Fast learning (dense rewards), high sample efficiency
- **Cons**: Low-level only (can't do waypoint nav)
- **Best for**: Fast response, acrobatics, tight control

**Level 3 (Attitude)**:
- **Pros**: Still relatively fast learning, more generalizable than L4
- **Cons**: Sparser rewards than L4
- **Best for**: Smooth flight, maneuvers

**Level 2 (HSA)**:
- **Pros**: High-level behaviors
- **Cons**: Very sparse rewards, slow learning
- **Best for**: Fuel optimization, loitering

**Level 1 (Waypoint)**:
- **Pros**: End-to-end navigation
- **Cons**: Extremely sparse rewards, very slow learning
- **Best for**: Path planning, obstacle avoidance

**Recommendation**: Start with L4 (Rate) for fastest results, then try higher levels.

### How do I compare my results to the paper?

**Reproduce baseline**:
```bash
# Run the exact same experiments
python examples/02_rl_vs_pid_demo.py --duration 10.0 --runs 50

# Use same hyperparameters
cd learned_controllers
python train_rate.py --config config/paper_hyperparams.yaml
```

**Metrics to report**:
- Settling time (within 5%)
- Overshoot percentage
- Steady-state error
- RMSE over entire episode
- Control smoothness (variance of control derivative)

### Can I use a different RL algorithm?

**Yes!** The code is designed to be algorithm-agnostic.

**Supported** (via Stable-Baselines3):
- PPO (with or without LSTM)
- SAC, TD3, A2C, DDPG
- DQN (for discrete action spaces)

**Example** (using SAC instead of PPO):
```python
from stable_baselines3 import SAC

model = SAC("MlpPolicy", env, ...)
model.learn(total_timesteps=1_000_000)
```

**Recommendation**: PPO with LSTM works well for flight control. SAC may be better for continuous, off-policy learning.

### How do I add a new sensor?

**Implement `SensorInterface`**:

```python
from interfaces import SensorInterface

class GPSSensor(SensorInterface):
    """GPS sensor with realistic noise."""

    def __init__(self, noise_std=1.0):
        self.noise_std = noise_std

    def sense(self, true_state: AircraftState) -> np.ndarray:
        """Add GPS noise to true position."""
        position = true_state.position
        noise = np.random.normal(0, self.noise_std, size=3)
        return position + noise

# Use in simulation
sensor = GPSSensor(noise_std=2.0)  # 2m standard deviation
measured_position = sensor.sense(state)
```

---

## Hardware & Deployment

### When will hardware support be added?

**Phase 6** (Hardware Interface) is planned but not yet started. See [Implementation Roadmap](design_docs/12_IMPLEMENTATION_ROADMAP.md).

**Estimated timeline**: 6-12 months depending on resources.

**What to expect**:
- Teensy 4.1 flight controller
- MAVLink protocol
- HIL (Hardware-in-the-Loop) testing
- Safety systems (failsafes, watchdogs)

### Can I help with hardware integration?

**Yes!** We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md#areas-for-contribution).

**Prerequisites**:
- Experience with embedded systems (Teensy/Arduino)
- Flight controller knowledge (MAVLink, betaflight, etc.)
- Access to test hardware

### How do I export RL model for deployment?

**ONNX export** (planned for Phase 8):

```python
import torch
from stable_baselines3 import PPO

# Load trained model
model = PPO.load("models/rate_controller_best.zip")

# Extract policy network
policy = model.policy

# Export to ONNX
dummy_input = torch.randn(1, obs_dim)
torch.onnx.export(
    policy,
    dummy_input,
    "rate_controller.onnx",
    input_names=['observation'],
    output_names=['action'],
    dynamic_axes={'observation': {0: 'batch_size'}}
)
```

**Then** use ONNX Runtime on embedded device (Jetson, RasPi, etc.)

---

## Still Have Questions?

### Where to ask?

- **GitHub Discussions**: General questions, ideas, research discussions
- **GitHub Issues**: Bug reports, feature requests

### How to get help effectively

**Good question**:
> "I'm trying to train a rate controller but getting NaN rewards after 10k steps.
> I'm using PPO with default hyperparameters on Ubuntu 20.04, Python 3.8.
> Here's my training log: [pastebin link]
> What might be wrong?"

**Bad question**:
> "It doesn't work. Help!"

**Include**:
1. What you're trying to do
2. What you expected
3. What actually happened
4. Your environment (OS, Python version)
5. Relevant code or logs
6. What you've already tried

---

## Common Error Messages

### "AssertionError: Rate agent expects RATE mode, got ATTITUDE"

**Cause**: Sending wrong control mode to agent

**Fix**: Match command mode to agent level
```python
# Wrong
agent = RateAgent(config)
command = ControlCommand(mode=ControlMode.ATTITUDE, ...)  # Wrong mode!

# Correct
agent = RateAgent(config)
command = ControlCommand(mode=ControlMode.RATE, ...)  # Correct!
```

### "ValueError: observation must be 18-dimensional, got 20"

**Cause**: Observation space mismatch in RL env

**Fix**: Check environment observation construction
```python
# Debug observation
obs = env.reset()
print(f"Observation shape: {obs.shape}")  # Should be (18,) for rate env
print(f"Observation: {obs}")
```

### "RuntimeError: CUDA out of memory"

**Cause**: GPU memory exhausted

**Fix**: See [Training crashes with CUDA OOM](#training-crashes-with-cuda-out-of-memory)

### "FileNotFoundError: config file not found"

**Cause**: Running script from wrong directory

**Fix**: Always run from project root
```bash
# Wrong
cd examples
python waypoint_mission.py

# Correct
cd /path/to/controls  # Project root
python examples/waypoint_mission.py
```

---

**Last Updated**: 2025-11-16

**Didn't find your question?** Ask in [GitHub Discussions](https://github.com/yourusername/controls/discussions)!
