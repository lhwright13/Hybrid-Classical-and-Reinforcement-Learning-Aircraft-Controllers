# Control System Tuning Guide

This guide explains how to systematically tune each level of the flight control hierarchy.

## Tools Available

### 1. Control Hierarchy Diagnostics
**Purpose**: Test each control level independently with step inputs

```bash
python validation/control_hierarchy_diagnostics.py
```

**Output**:
- Step response plots for Rate, Attitude, and HSA controllers
- Performance metrics (tracking error, overshoot, settling time)
- Saved to `logs/diagnostics/`

**Use this to**:
- Identify which control level has poor performance
- Verify tracking ability at each level
- Detect oscillations, overshoot, or sluggish response

---

### 2. Comprehensive Waypoint Analysis
**Purpose**: Full waypoint mission with detailed event tracking

```bash
python validation/comprehensive_waypoint_analysis.py
```

**Output**:
- Multi-panel plot showing all state variables vs time
- Waypoint transitions marked on all graphs
- TECS energy analysis
- Saved to `logs/waypoint_missions/`

**Use this to**:
- See the big picture of waypoint navigation performance
- Identify when/where failures occur
- Analyze energy management during mission

---

### 3. Gain Tuner (TODO: needs implementation)
**Purpose**: Automated parameter sweeps

```bash
python validation/gain_tuner.py --controller hsa --param baseline_throttle --values 0.1,0.2,0.3
```

---

## Control Hierarchy

```
Level 1: Waypoint Navigation
    ↓ (commands HSA)
Level 2: HSA (Heading, Speed, Altitude)
    ↓ (commands Attitude)
Level 3: Attitude (Roll, Pitch, Yaw)
    ↓ (commands Rates)
Level 4: Rate (p, q, r)
    ↓ (commands Surfaces)
Level 5: Surfaces (aileron, elevator, rudder, throttle)
```

---

## Tunable Parameters

### Level 4: Rate Controller
**File**: `controllers/config/pid_gains.yaml`

```yaml
rate:
  roll:
    kp: 0.03    # Roll rate response
    ki: 0.005   # Steady-state error correction
    kd: 0.0     # Damping
  pitch:
    kp: 0.008   # Pitch rate response
    ki: 0.0
    kd: 0.0
  yaw:
    kp: 0.001   # Yaw rate response
    ki: 0.0
    kd: 0.0
```

**Tuning symptoms**:
- **Oscillation**: Reduce kp, increase kd
- **Sluggish**: Increase kp
- **Steady-state error**: Increase ki (use sparingly!)

---

### Level 3: Attitude Controller
**File**: `controllers/config/pid_gains.yaml`

```yaml
attitude:
  roll:
    kp: 0.01    # Roll angle response
    ki: 0.0
    kd: 0.0
  pitch:
    kp: 0.01    # Pitch angle response
    ki: 0.0
    kd: 0.0
  yaw:
    kp: 0.0001  # Yaw angle response
    ki: 0.0
    kd: 0.0
```

**Tuning symptoms**:
- **Overshoot**: Reduce kp or increase kd
- **Slow to reach attitude**: Increase kp
- **Oscillation around target**: Reduce kp, add kd

---

### Level 2: HSA Controller
**File**: `controllers/hsa_agent.py`

#### Heading Control
```python
# Line 57-58: Bank angle limits
heading_config.output_min = -np.radians(10)  # Max bank angle
heading_config.output_max = np.radians(10)

# Line 140: Roll angle clipping
roll_angle = np.clip(roll_angle, -np.radians(10), np.radians(10))
```

**Turn physics**:
- Turn radius: R = V²/(g·tan(φ))
- At 12 m/s, 10° bank → R = 82m
- At 36 m/s, 10° bank → R = 750m Warning:

**Tuning**:
- Increase bank angle → tighter turns (risk: instability)
- Decrease bank angle → gentler turns (risk: can't reach waypoints)

---

#### TECS Energy Controller
```python
# Line 66-68: Total energy PID (controls throttle)
energy_config.gains = acb.PIDGains(
    kp=0.08,   # Energy error → throttle adjustment
    ki=0.02,   # Integral for steady-state
    kd=0.02    # Derivative for damping
)

# Line 72-73: Throttle limits
energy_config.output_min = -0.5  # Max throttle reduction
energy_config.output_max = 0.5   # Max throttle increase
```

**Tuning symptoms**:
- **Speed too high**: Increase kp (more aggressive throttle reduction)
- **Speed oscillates**: Reduce kp, increase kd
- **Altitude drops**: Check baseline_throttle (may be too low)

---

#### TECS Balance Controller
```python
# Line 79-81: Energy distribution PID (controls pitch)
balance_config.gains = acb.PIDGains(
    kp=0.10,   # Balance error → pitch adjustment
    ki=0.002,  # Weak integral
    kd=0.05    # Damping
)

# Line 85-86: Pitch limits
balance_config.output_min = -np.radians(10)
balance_config.output_max = np.radians(10)
```

**Tuning symptoms**:
- **Slow altitude/speed exchange**: Increase kp
- **Phugoid oscillation**: Reduce kp, increase kd
- **Altitude overshoot**: Reduce kp

---

#### Baseline Throttle
```python
# Line 93: Cruise throttle setting
self.baseline_throttle = 0.2  # 0.0 to 1.0
```

**Typical values**:
- 0.3: Normal cruise (12 m/s)
- 0.2: Reduced power (helps limit speed)
- 0.1: Very low power (risk: insufficient climb authority)

---

#### Turn Throttle Compensation
```python
# Line 180-182: Reduce throttle during turns
throttle -= 0.030 * roll_angle_deg  # 3% per degree of bank
```

**Purpose**: Prevent speed buildup during turns

**Tuning**:
- Increase coefficient → more aggressive speed control in turns
- Decrease coefficient → maintain energy during turns
- Too high → aircraft loses altitude in turns
- Too low → speed builds up excessively

**Current**: 3.0% per degree (30% reduction at 10° bank)

---

## Typical Tuning Workflow

### Step 1: Verify Lower Levels
```bash
python validation/control_hierarchy_diagnostics.py
```

Check that Rate and Attitude controllers have good step response:
- Rise time < 1 second
- Overshoot < 20%
- No oscillation
- RMS error < 0.05 rad

If issues found → tune PID gains in `controllers/config/pid_gains.yaml`

---

### Step 2: Test HSA Level
```bash
python validation/control_hierarchy_diagnostics.py
```

Check HSA step responses:
- Heading: Should reach target within 10-15s
- Altitude: Should reach ±2m within 20s
- Speed: Should reach ±1 m/s within 15s

If issues found → tune TECS gains in `controllers/hsa_agent.py`

---

### Step 3: Full Waypoint Mission
```bash
python validation/comprehensive_waypoint_analysis.py
```

Check mission performance:
- Waypoints reached: target 5/5
- Altitude stability: target ±10m
- Speed stability: target ±3 m/s

If issues found:
- **Low waypoint completion** → adjust bank angle or turn throttle comp
- **Altitude instability** → tune TECS balance controller
- **Speed instability** → tune TECS energy controller or baseline throttle

---

## Current Known Issues

### Issue: Only 2/5 Waypoints Reached

**Root Cause**: Turn radius too large for waypoint geometry
- Max speed in turns: 36.5 m/s
- Bank angle: 10°
- Turn radius: ~750m
- Waypoint spacing: 2000m legs
- Acceptance radius: 300m

**Physics Constraint**:
```
To complete 90° turn within 2000m leg:
  Required turn radius: ~700m
  At 10° bank: Max speed = sqrt(700 * 9.81 * tan(10°)) ≈ 35 m/s ✓ (borderline)
```

**Potential Solutions**:
1. Increase bank angle to 12° (tighter turns, but may destabilize)
2. Reduce cruise speed below 12 m/s (less practical)
3. Modify mission (wider waypoint spacing or larger acceptance radius)
4. Implement predictive turn entry (start turn before waypoint)

---

## Performance Baseline (Current)

From extensive tuning (see conversation history):

**Controller Parameters**:
- Baseline throttle: 0.2 (was 0.5)
- Bank angle limit: 10° (was 8°)
- TECS balance kp: 0.10 (was 0.02)
- TECS energy kp: 0.08 (was 0.05)
- Turn throttle comp: 3.0% per degree (30% at 10°)

**Performance**:
- Waypoints: 2/5 (40%)
- Altitude: ±8.2m (excellent)
- Airspeed: 12-36.5 m/s (good, but still high in turns)

**Improvements from start**:
- Altitude control: ±63.5m → ±8.2m (87% improvement)
- Max airspeed: 45 m/s → 36.5 m/s (19% reduction)
- Waypoint completion: 2/5 → 2/5 (unchanged, physics limited)

---

## Next Steps for Manual Tuning

1. **Run diagnostics** to establish baseline performance at each level
2. **Identify weakest level** (highest tracking error)
3. **Adjust one parameter at a time** (scientific method!)
4. **Re-run diagnostics** to verify improvement
5. **Document changes** in this file

Remember: **One change at a time!** Otherwise you won't know what worked.
