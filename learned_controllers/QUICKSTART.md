# Quick Start Guide - Learned Rate Controller

This guide will get you up and running with the learned rate controller in 5 minutes.

## Step 1: Install Dependencies

Make sure your virtual environment is activated and install the RL dependencies:

```bash
cd /Users/lhwri/controls
source venv/bin/activate
pip install stable-baselines3[extra] sb3-contrib gymnasium tensorboard
```

## Step 2: Run Examples

Test that everything works:

```bash
./venv/bin/python learned_controllers/example_usage.py
```

You should see output showing:
- Environment interaction
- Command generation
- Flight envelope sampling
- Reward function behavior
- Training configuration

## Step 3: Train Your First Model (Quick Test)

Let's do a quick 10k step training run to verify everything works:

```bash
# Create a quick test config
cat > learned_controllers/config/quick_test.yaml << 'EOF'
environment:
  difficulty: "easy"
  episode_length: 5.0
  dt: 0.02
  command_type: "step"

training:
  total_timesteps: 10000  # Quick test
  n_envs: 2
  eval_freq: 5000
  save_freq: 5000
  log_interval: 1

curriculum:
  enabled: false

ppo:
  learning_rate: 3.0e-4
  n_steps: 512
  batch_size: 64
  n_epochs: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  clip_range_vf: null
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  use_sde: false
  sde_sample_freq: -1

lstm:
  enabled: true
  lstm_hidden_size: 128
  n_lstm_layers: 1
  features_dim: 64

mlp:
  net_arch: [128, 64]

normalize:
  obs: false
  reward: false

paths:
  model_save_dir: "learned_controllers/models/checkpoints"
  tensorboard_log: "learned_controllers/logs/tensorboard"
  best_model_path: "learned_controllers/models/test_model"

evaluation:
  n_eval_episodes: 3
  deterministic: true
  render: false

seed: 42
EOF

# Run quick training
./venv/bin/python learned_controllers/train_rate.py \
  --config learned_controllers/config/quick_test.yaml
```

This should complete in 1-2 minutes. Watch for output showing:
- Environment creation
- PPO model initialization
- Training episodes
- Evaluation results

## Step 4: Check Training Progress

While training (or after), view TensorBoard:

```bash
source venv/bin/activate
tensorboard --logdir learned_controllers/logs/tensorboard
```

Then open http://localhost:6006 in your browser.

Look for:
- `rollout/ep_rew_mean` - Episode reward (should increase)
- `train/loss` - Training loss
- `eval/mean_reward` - Evaluation performance

## Step 5: Evaluate the Trained Model

After training completes:

```bash
./venv/bin/python learned_controllers/eval_rate.py \
  --model learned_controllers/models/test_model.zip \
  --n-episodes 5 \
  --difficulty easy
```

This will show:
- Settling time for each axis
- Overshoot percentages
- Tracking RMSE
- Control smoothness
- Success rate

## Step 6: Full Training (1M Steps)

Once you've verified everything works, run full training:

```bash
# This will take several hours (1M steps with curriculum)
./venv/bin/python learned_controllers/train_rate.py

# Monitor progress
tensorboard --logdir learned_controllers/logs/tensorboard
```

The full training uses curriculum learning:
1. **Easy phase** (300k steps): Simple single-axis commands
2. **Medium phase** (400k steps): Multi-axis commands
3. **Hard phase** (300k steps): Aggressive random walk commands

## Step 7: Compare with PID Baseline

After full training:

```bash
./venv/bin/python learned_controllers/eval_rate.py \
  --model learned_controllers/models/best_rate_controller.zip \
  --n-episodes 20 \
  --difficulty medium \
  --compare-pid
```

This will run both controllers and show a comparison table.

## Troubleshooting

### Error: "No module named 'stable_baselines3'"
```bash
source venv/bin/activate
pip install stable-baselines3[extra] sb3-contrib gymnasium
```

### Error: "CUDA out of memory"
Training uses CPU by default. If you have GPU issues, this shouldn't happen. If it does, the model will automatically fall back to CPU.

### Training is slow
- Reduce `n_envs` in config (default: 4 → try 2)
- Reduce `n_steps` (default: 2048 → try 1024)
- Use smaller LSTM (hidden_size: 256 → 128, layers: 2 → 1)

### Reward not improving
- Check TensorBoard for `rollout/ep_rew_mean`
- Verify environment is working: run `example_usage.py`
- Try easier difficulty first
- Increase training time

## Expected Results

After full training (1M steps), you should see:

**Learned Controller:**
- Settling time: 0.3-0.5s
- Overshoot: 5-15%
- Tracking RMSE: <0.1 rad/s
- Success rate: >90%

**vs PID Baseline:**
- Similar or better settling time
- Potentially lower overshoot (smoother)
- Similar tracking accuracy

## Next Steps

Once you have a trained model:

1. **Deploy in control system**: Use `LearnedRateAgent` in your flight controller
2. **Experiment with architectures**: Try MLP vs LSTM, different sizes
3. **Tune rewards**: Adjust reward function weights for your priorities
4. **Train other controllers**: Apply same approach to attitude, HSA levels

## Getting Help

- Check the full README: `learned_controllers/README.md`
- Review the implementation plan: `learned_controllers/plan.md`
- Run examples: `./venv/bin/python learned_controllers/example_usage.py`
- Check code documentation in each module

Happy training! 🚁
