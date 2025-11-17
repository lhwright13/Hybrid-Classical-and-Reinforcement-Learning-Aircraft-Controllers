# Testing the Learned Rate Controller

## Quick Start

### 1. Launch Interactive GUI
```bash
./venv/bin/python examples/launch_pygame_gui_with_learned_rate.py
```

**What it does:**
- Loads your trained RL rate controller
- Creates pygame GUI for manual flight testing
- Allows real-time toggle between Learned and PID controllers
- Logs telemetry to HDF5 file

**Controls:**
- **`L` key** - Toggle between Learned RL and PID rate controllers
- **`R` key** - Reset simulation
- **`D` key** - Toggle debug panel
- **Joystick** - Command pitch/roll rates (in Rate mode)
- **Sliders** - Throttle and rudder

### 2. Evaluate Offline
```bash
./venv/bin/python learned_controllers/eval_rate.py \
  --model learned_controllers/models/best_rate_controller/best_model.zip \
  --n-episodes 10 \
  --difficulty medium \
  --compare-pid
```

**What it does:**
- Runs learned controller on predefined scenarios
- Compares with PID baseline
- Outputs metrics table

## Files

### GUI Testing
- **`launch_pygame_gui_with_learned_rate.py`** - Main interactive test launcher
- **`../gui/simulation_worker_learned.py`** - Simulation worker with toggle support

### Offline Testing
- **`../learned_controllers/eval_rate.py`** - Automated evaluation script
- **`../learned_controllers/eval/metrics.py`** - Metrics calculation

### Documentation
- **`../learned_controllers/TEST_PLAN.md`** - Comprehensive test plan
- **`../learned_controllers/TESTING_SUMMARY.md`** - Quick summary & results

## Command-Line Options

### GUI Launcher
```bash
./venv/bin/python examples/launch_pygame_gui_with_learned_rate.py \
  --model PATH_TO_MODEL.zip \
  --log telemetry.h5
```

### Evaluator
```bash
./venv/bin/python learned_controllers/eval_rate.py \
  --model PATH_TO_MODEL.zip \
  --n-episodes 10 \
  --difficulty medium \
  --compare-pid \
  --episode-length 10.0 \
  --command-type step
```

## Testing Workflow

1. **Offline Evaluation** (5 min)
   - Run eval script to get baseline metrics
   - Check settling time, overshoot, tracking error

2. **Interactive Testing** (15 min)
   - Launch GUI
   - Switch to RATE mode
   - Fly with learned controller
   - Press `L` to toggle to PID
   - Feel the difference!

3. **Data Analysis** (optional)
   - Load HDF5 telemetry file
   - Plot tracking performance
   - Compare controller types

## Expected Results

### Learned Controller
- ⚡ **90%+ faster settling time** than PID
- Fast, aggressive response
- Some overshoot on aggressive inputs
- Higher tracking error at steady-state

### PID Controller
- Smooth, predictable response
- No overshoot
- Better steady-state accuracy
- Slower settling time

## Tips

- **Start with gentle inputs** to get a feel for each controller
- **Toggle mid-flight** to immediately compare responsiveness
- **Watch telemetry** to see which controller is active
- **Log data** for later analysis (automatic)

## Troubleshooting

### Model not loading
- Check path: `ls learned_controllers/models/best_rate_controller/`
- Try different model: `--model learned_controllers/models/checkpoints/final_model.zip`

### No visual difference
- Ensure you're in RATE mode (not Attitude/HSA)
- Check telemetry shows controller switching
- Try more aggressive inputs

### Instability
- Reset simulation (`R` key)
- Reduce joystick inputs
- Try easier difficulty in eval mode

## Next Steps

After testing:
1. Review `learned_controllers/TESTING_SUMMARY.md` for results
2. Decide on retraining needs (if any)
3. Deploy in your system or retrain with adjusted rewards
4. Consider hybrid approach (Learned + PID)

Happy testing! 🚁
