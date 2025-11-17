# Learned Rate Controller

This directory contains the infrastructure for training and deploying RL-based rate controllers as an alternative to classical PID control.

## Overview

The learned rate controller uses **PPO (Proximal Policy Optimization) with LSTM** to learn rate tracking behavior from scratch through reinforcement learning. The trained policy can replace the PID-based `RateAgent` in the control hierarchy.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `stable-baselines3[extra]` - PPO implementation
- `sb3-contrib` - RecurrentPPO for LSTM policies
- `gymnasium` - RL environment interface
- `tensorboard` - Training visualization

### 2. Train a Model

```bash
# Make sure you're in the project root directory
cd /path/to/controls

# Train with default settings (LSTM + curriculum learning)
./venv/bin/python learned_controllers/train_rate.py

# OR use the helper script
./learned_controllers/run.sh learned_controllers/train_rate.py

# Train with MLP (no LSTM)
./venv/bin/python learned_controllers/train_rate.py --no-lstm

# Use custom config
./venv/bin/python learned_controllers/train_rate.py --config my_config.yaml
```

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir learned_controllers/logs/tensorboard
```

### 3. Evaluate Performance

```bash
# Evaluate trained model
./venv/bin/python learned_controllers/eval_rate.py \
  --model learned_controllers/models/best_rate_controller.zip \
  --n-episodes 10 \
  --difficulty medium

# Compare with PID baseline
./venv/bin/python learned_controllers/eval_rate.py \
  --model learned_controllers/models/best_rate_controller.zip \
  --compare-pid
```

### 4. Deploy in Control System

```python
from controllers.learned_rate_agent import LearnedRateAgent
from controllers.config import load_config

# Load trained model
config = load_config()
agent = LearnedRateAgent(
    model_path="learned_controllers/models/best_rate_controller.zip",
    config=config,
    fallback_to_pid=True,  # PID fallback on model failure
)

# Use like any other agent
command = ControlCommand(mode=ControlMode.RATE, ...)
surfaces = agent.compute_action(command, state)
```

## Architecture

### Observation Space (18-dim)
- Current rates: `[p, q, r]` (rad/s)
- Commanded rates: `[p_cmd, q_cmd, r_cmd]` (rad/s)
- Rate errors: `[p_err, q_err, r_err]` (rad/s)
- Flight state: `[airspeed, altitude, roll, pitch, yaw]`
- Previous action: `[aileron, elevator, rudder, throttle]`

### Action Space (4-dim continuous)
- Aileron: `[-1, 1]`
- Elevator: `[-1, 1]`
- Rudder: `[-1, 1]`
- Throttle: `[0, 1]`

### Reward Function
Multi-component reward:
- **Tracking**: `-MSE(p_err, q_err, r_err)` (primary objective)
- **Smoothness**: `-|Δu|` (penalize chattering)
- **Stability**: Bonus for safe flight envelope
- **Oscillation**: Penalty for excessive oscillation

See `envs/rewards.py` for details.

### Network Architecture
**LSTM Policy** (default):
- Input embedding: Linear(18 → 128) + ReLU
- LSTM layers: 2 layers, 256 hidden units each
- Actor head: Linear(256 → 128 → 64 → 4)
- Critic head: Linear(256 → 128 → 64 → 1)

**MLP Policy** (alternative):
- Shared layers: [256, 128, 64]
- Separate actor and critic heads

## Configuration

Training is configured via `config/ppo_lstm.yaml`:

```yaml
# Environment settings
environment:
  difficulty: "easy"
  episode_length: 10.0
  dt: 0.02
  command_type: "step"

# PPO hyperparameters
ppo:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 64
  ...

# Curriculum learning
curriculum:
  enabled: true
  phases:
    - name: "easy"
      difficulty: "easy"
      timesteps: 300000
    ...
```

## Curriculum Learning

Training progresses through 3 phases:
1. **Easy** (300k steps): Single-axis, small commands, high altitude
2. **Medium** (400k steps): Multi-axis, moderate commands, varying altitude
3. **Hard** (300k steps): Aggressive maneuvers, coupled dynamics, random walk commands

## Evaluation Metrics

The evaluation script computes:
- **Settling time**: Time to reach ±5% of commanded rate
- **Overshoot**: Maximum deviation beyond command (%)
- **Rise time**: Time to 90% of command
- **Steady-state error**: Mean error after settling (rad/s)
- **Tracking RMSE**: Root mean squared error
- **Control smoothness**: Mean absolute action change
- **Success rate**: % of episodes settled within time limit

## File Structure

```
learned_controllers/
├── README.md              # This file
├── plan.md                # Detailed implementation plan
├── envs/
│   ├── rate_env.py        # Gymnasium environment
│   └── rewards.py         # Reward functions
├── networks/
│   └── lstm_policy.py     # LSTM policy architecture
├── models/
│   ├── checkpoints/       # Training checkpoints
│   └── best_rate_controller.zip  # Best model
├── data/
│   └── generators.py      # Command generators
├── eval/
│   └── metrics.py         # Evaluation metrics
├── config/
│   └── ppo_lstm.yaml      # Training config
├── train_rate.py          # Training script
└── eval_rate.py           # Evaluation script
```

## Tips and Tricks

### Training Tips
1. **Start with curriculum learning**: Progressively harder tasks improve final performance
2. **Monitor TensorBoard**: Watch for reward improvement and stability
3. **Tune reward weights**: Balance tracking vs smoothness vs stability
4. **Try different architectures**: LSTM for temporal patterns, MLP for speed

### Deployment Tips
1. **Enable PID fallback**: Graceful degradation on model failures
2. **Test thoroughly**: Evaluate across difficulty levels before deployment
3. **Monitor performance**: Track metrics during real flights
4. **Periodic retraining**: Collect real-world data and fine-tune

## Troubleshooting

**Issue**: Training reward not improving
- **Solution**: Check reward function weights, reduce difficulty, increase training time

**Issue**: High overshoot or oscillation
- **Solution**: Increase smoothness penalty weight, reduce learning rate

**Issue**: Model fails to load
- **Solution**: Check model path, ensure SB3 versions match training/deployment

**Issue**: Poor generalization to new conditions
- **Solution**: Increase training diversity, use harder curriculum phases

## Next Steps

After training a rate controller, you can:
1. **Deploy in real system**: Replace PID rate agent with learned agent
2. **Train higher-level controllers**: Use same approach for attitude, HSA, etc.
3. **Hybrid control**: Combine RL and PID (e.g., RL outputs corrections to PID)
4. **Domain adaptation**: Fine-tune on real aircraft data

## References

- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [RecurrentPPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
