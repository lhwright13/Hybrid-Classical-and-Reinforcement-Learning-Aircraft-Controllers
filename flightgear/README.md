# FlightGear Integration for Python Flight Control System

This directory contains scripts to visualize your Python flight simulation in FlightGear.

## Overview

FlightGear is used **ONLY for 3D visualization** - all physics are computed by your Python simulation. FlightGear acts as a renderer via its telnet property server, displaying the aircraft position/attitude that your simulation computes.

## Quick Start

### 1. Install FlightGear (if not already installed)

```bash
brew install --cask flightgear
```

### 2. Launch FlightGear with telnet enabled

```bash
open -a FlightGear --args --aircraft=c172p --airport=KSFO --telnet=5401
```

Or click "Fly" in the GUI, FlightGear will start at KSFO.

### 3. Run Python Simulation (in separate terminal)

```bash
cd /Users/lhwri/controls
python3 -u flightgear/fly_with_visualization.py
```

The script will:
- Connect to FlightGear on port 5401
- Freeze FlightGear's physics engine
- Run waypoint mission with Simplified 6DOF physics
- Update FlightGear display in real-time

## How It Works

### Architecture

```
┌─────────────────────────────────────┐
│   Python Simulation                 │
│   ─────────────────                 │
│   - Simplified 6DOF Physics         │
│   - Waypoint Navigation             │
│   - PID Controllers                 │
│   - All flight dynamics             │
└──────────┬──────────────────────────┘
           │ Telnet (port 5401)
           │ Position/Attitude commands
           ▼
┌─────────────────────────────────────┐
│   FlightGear (Physics FROZEN)       │
│   ──────────────────────────        │
│   - 3D Visualization ONLY           │
│   - Property server (telnet)        │
│   - Renders aircraft position       │
└─────────────────────────────────────┘
```

### Communication Protocol

Uses FlightGear's telnet property server to directly set aircraft state:

**Commands sent from Python:**
```
set /position/latitude-deg 37.6177
set /position/longitude-deg -122.3750
set /position/altitude-ft 328.084
set /orientation/roll-deg 0.0
set /orientation/pitch-deg 3.0
set /orientation/heading-deg 0.0
set /velocities/airspeed-kt 54.5
```

### Data Flow

1. Python simulation computes aircraft state using 6DOF dynamics
2. Python sends position/attitude commands via telnet
3. FlightGear updates aircraft position (physics frozen)
4. Loop repeats every timestep (100 Hz simulation, display updated as fast as possible)

## Files

- **`fly_with_visualization.py`**: Main script - fly waypoint mission with visualization
- **`README.md`**: This file

## Troubleshooting

### FlightGear doesn't start

Make sure FlightGear is installed:
```bash
open -a FlightGear  # Should launch FlightGear GUI
```

### Python can't connect (port 5401)

Launch FlightGear with telnet enabled:
```bash
open -a FlightGear --args --telnet=5401
```

Or set it in FlightGear GUI settings before clicking "Fly".

### Red screen in FlightGear

Aircraft is underground (altitude control issue). This happens with JSBSim physics. Use Simplified 6DOF instead (current default).

### Aircraft doesn't move in FlightGear

1. Check Python script shows "✓ Connected to FlightGear!"
2. Check Python script shows "✓ FlightGear physics disabled"
3. Verify waypoints are being reached in terminal output

## Examples

### Basic waypoint mission (current script)

```bash
python3 -u flightgear/fly_with_visualization.py
```

Flies a square pattern:
- WP1: (0, 0, 100m) - Start
- WP2: (2000, 0, 100m) - 2km north
- WP3: (2000, 2000, 100m) - 2km east
- WP4: (0, 2000, 100m) - 2km south
- WP5: (0, 0, 100m) - Back to start

**Expected Results (Simplified 6DOF):**
- Waypoints reached: 3/5
- Altitude hold: Good during straight flight (100m)
- Issues: Altitude loss during aggressive turns

## Technical Notes

### Why telnet instead of UDP native-fdm?

Telnet property server is simpler and more direct:
- No protocol XML needed
- Direct property access
- Can freeze FlightGear physics with single command
- More reliable connection

### Coordinate Systems

- **Python (NED)**: North-East-Down frame, altitude in meters AGL
- **FlightGear**: Geodetic lat/lon, altitude in feet MSL

The script converts NED to geodetic using flat-earth approximation around KSFO (37.6177°N, 122.3750°W).

### Physics Backends

**Simplified 6DOF** (current, works well):
- Fast, stable altitude control
- Reaches 3/5 waypoints
- Some altitude loss during turns
- Good for controller development

**JSBSim** (realistic but challenging):
- High-fidelity C172P model
- Requires careful PID tuning
- Currently has altitude control issues
- Needs elevator trim and better gains

### Performance

- Simulation runs at 100 Hz (0.01s timestep)
- Visualization updates every frame
- Real-time performance on modern hardware

## Next Steps

- [ ] Fix JSBSim altitude control (requires PID retuning + trim)
- [ ] Improve waypoint navigation (currently 3/5, target 5/5)
- [ ] Add learned controllers (RL agents)
- [ ] Export flight data for replay
- [ ] Add camera chase view option
