# Flight Controller - Classical Controller Implementation

## Overview

This document specifies the implementation of classical PID-based flight controllers for all 4 control levels. These controllers serve as:
1. **Baselines** for RL agent comparison
2. **Safety fallbacks** for hybrid agents
3. **Building blocks** extracted from dRehmFlight

## dRehmFlight Integration

### Code Extraction Strategy

We extract proven PID algorithms from dRehmFlight and adapt them to our multi-level architecture.

**Source Files** (from dRehmFlight):
- `controlANGLE()` - Angle mode PID
- `controlRATE()` - Rate mode PID
- `controlMixer()` - Vehicle-specific mixing
- PID gains and tuning parameters

**Adaptation Strategy**:
1. Extract C++ PID core (performance-critical)
2. Wrap in Pybind11 for Python access
3. Create Python controllers for Levels 1-3
4. Keep Level 4 (PID + Mixer) in C++

## Level 4: PID Controllers & Mixer (C++)

### PID Controller Implementation

**File**: `core/pid_controller.cpp`

```cpp
class PIDController {
public:
    PIDController(float kp, float ki, float kd, float i_limit);
    
    float compute(float setpoint, float measurement, float dt);
    void reset();
    
private:
    float kp_, ki_, kd_;
    float i_limit_;
    float integral_;
    float prev_error_;
};

float PIDController::compute(float setpoint, float measurement, float dt) {
    float error = setpoint - measurement;
    
    // Integral with anti-windup
    integral_ += error * dt;
    integral_ = constrain(integral_, -i_limit_, i_limit_);
    
    // Derivative
    float derivative = (error - prev_error_) / dt;
    
    // PID output
    float output = kp_ * error + ki_ * integral_ + kd_ * derivative;
    
    prev_error_ = error;
    return output;
}
```

### Attitude Control (C++)

**File**: `core/attitude_control.cpp`

```cpp
class AttitudeController {
public:
    AttitudeController(const PIDGains& roll, const PIDGains& pitch, const PIDGains& yaw);
    
    ControlOutput compute(
        const AttitudeState& state,
        const AttitudeCommand& command,
        float dt
    );
    
private:
    PIDController roll_pid_;
    PIDController pitch_pid_;
    PIDController yaw_pid_;
};

ControlOutput AttitudeController::compute(
    const AttitudeState& state,
    const AttitudeCommand& command,
    float dt
) {
    ControlOutput output;
    
    // Roll control
    float roll_error = command.roll - state.roll;
    float roll_rate_cmd = roll_pid_.compute(roll_error, state.p, dt);
    
    // Pitch control
    float pitch_error = command.pitch - state.pitch;
    float pitch_rate_cmd = pitch_pid_.compute(pitch_error, state.q, dt);
    
    // Yaw control (rate mode)
    float yaw_error = command.yaw_rate - state.r;
    output.yaw = yaw_pid_.compute(yaw_error, 0.0, dt);
    
    // Rate loop (inner loop)
    output.roll = roll_rate_cmd;
    output.pitch = pitch_rate_cmd;
    
    return output;
}
```

### Control Mixer

**File**: `core/control_mixer.cpp`

```cpp
class ControlMixer {
public:
    virtual ~ControlMixer() = default;
    virtual SurfaceCommands mix(const ControlOutput& control) = 0;
};

// Fixed-wing mixer
class FixedWingMixer : public ControlMixer {
public:
    SurfaceCommands mix(const ControlOutput& control) override {
        SurfaceCommands surfaces;
        
        surfaces.elevator = control.pitch;  // Pitch → elevator
        surfaces.aileron = control.roll;    // Roll → aileron
        surfaces.rudder = control.yaw;      // Yaw → rudder
        surfaces.throttle = control.throttle;
        
        // Saturate
        surfaces.elevator = constrain(surfaces.elevator, -1.0f, 1.0f);
        surfaces.aileron = constrain(surfaces.aileron, -1.0f, 1.0f);
        surfaces.rudder = constrain(surfaces.rudder, -1.0f, 1.0f);
        surfaces.throttle = constrain(surfaces.throttle, 0.0f, 1.0f);
        
        return surfaces;
    }
};

// Quadrotor mixer
class QuadrotorMixer : public ControlMixer {
public:
    SurfaceCommands mix(const ControlOutput& control) override {
        SurfaceCommands surfaces;
        
        // + configuration
        float m1 = control.throttle - control.roll - control.pitch - control.yaw;
        float m2 = control.throttle - control.roll + control.pitch + control.yaw;
        float m3 = control.throttle + control.roll + control.pitch - control.yaw;
        float m4 = control.throttle + control.roll - control.pitch + control.yaw;
        
        // Saturate
        surfaces.motor1 = constrain(m1, 0.0f, 1.0f);
        surfaces.motor2 = constrain(m2, 0.0f, 1.0f);
        surfaces.motor3 = constrain(m3, 0.0f, 1.0f);
        surfaces.motor4 = constrain(m4, 0.0f, 1.0f);
        
        return surfaces;
    }
};
```

### Pybind11 Bindings

**File**: `core/bindings.cpp`

```cpp
#include <pybind11/pybind11.h>
#include "pid_controller.h"
#include "attitude_control.h"
#include "control_mixer.h"

namespace py = pybind11;

PYBIND11_MODULE(controls_core, m) {
    py::class_<PIDController>(m, "PIDController")
        .def(py::init<float, float, float, float>())
        .def("compute", &PIDController::compute)
        .def("reset", &PIDController::reset);
    
    py::class_<AttitudeController>(m, "AttitudeController")
        .def(py::init<const PIDGains&, const PIDGains&, const PIDGains&>())
        .def("compute", &AttitudeController::compute);
    
    py::class_<ControlMixer>(m, "ControlMixer");
    
    py::class_<FixedWingMixer, ControlMixer>(m, "FixedWingMixer")
        .def(py::init<>())
        .def("mix", &FixedWingMixer::mix);
    
    py::class_<QuadrotorMixer, ControlMixer>(m, "QuadrotorMixer")
        .def(py::init<>())
        .def("mix", &QuadrotorMixer::mix);
}
```

## Level 3: Attitude Controller (Python)

**File**: `controllers/level3_attitude.py`

```python
import numpy as np
from controls_core import AttitudeController  # C++ import
from controllers.types import ControlCommand, AircraftState, ControlMode


class Level3AttitudeController:
    """Level 3: Stick & Throttle to Rate commands."""
    
    def __init__(self, config):
        self.config = config
        
        # Use C++ PID controller
        self.attitude_controller = AttitudeController(
            config["roll_gains"],
            config["pitch_gains"],
            config["yaw_gains"]
        )
        
        # Stick scaling
        self.max_roll_angle = np.radians(config.get("max_roll", 45))
        self.max_pitch_angle = np.radians(config.get("max_pitch", 30))
        self.max_yaw_rate = np.radians(config.get("max_yaw_rate", 90))
    
    def compute_command(
        self,
        state: AircraftState,
        input_command: ControlCommand,
        dt: float
    ) -> ControlCommand:
        """Convert stick inputs to rate commands."""
        
        # Map stick to desired angles
        roll_des = input_command.roll_cmd * self.max_roll_angle
        pitch_des = input_command.pitch_cmd * self.max_pitch_angle
        yaw_rate_des = input_command.yaw_cmd * self.max_yaw_rate
        
        # Create attitude command for C++ controller
        attitude_cmd = {
            "roll": roll_des,
            "pitch": pitch_des,
            "yaw_rate": yaw_rate_des,
            "throttle": input_command.throttle
        }
        
        # Compute PID output (C++)
        output = self.attitude_controller.compute(state, attitude_cmd, dt)
        
        # Return rate commands for Level 4
        return ControlCommand(
            mode=ControlMode.SURFACE,
            elevator=output.pitch,
            aileron=output.roll,
            rudder=output.yaw,
            throttle=input_command.throttle
        )
    
    def reset(self):
        self.attitude_controller.reset()
```

## Level 2: HSA Controller (Python)

**File**: `controllers/level2_hsa.py`

```python
class Level2HSAController:
    """Level 2: HSA (Heading, Speed, Altitude) to Attitude commands."""
    
    def __init__(self, config):
        self.config = config
        
        # PID controllers for each HSA component
        self.heading_pid = PIDController(**config["heading_gains"])
        self.speed_pid = PIDController(**config["speed_gains"])
        self.altitude_pid = PIDController(**config["altitude_gains"])
    
    def compute_command(
        self,
        state: AircraftState,
        input_command: ControlCommand,
        dt: float
    ) -> ControlCommand:
        """Convert HSA commands to attitude commands."""
        
        # Heading control → roll command (coordinated turn)
        heading_error = self._wrap_angle(
            input_command.heading - state.heading
        )
        yaw_rate_cmd = self.heading_pid.compute(heading_error, 0.0, dt)
        roll_cmd = np.clip(
            yaw_rate_cmd * 0.5,  # Convert yaw rate to roll
            -np.radians(30),
            np.radians(30)
        )
        
        # Speed control → throttle command
        speed_error = input_command.speed - state.airspeed
        throttle_cmd = 0.5 + self.speed_pid.compute(speed_error, 0.0, dt)
        throttle_cmd = np.clip(throttle_cmd, 0, 1)
        
        # Altitude control → pitch command
        altitude_error = input_command.altitude - state.altitude
        climb_rate_cmd = np.clip(altitude_error * 0.1, -5, 5)  # P on altitude
        pitch_cmd = self.altitude_pid.compute(
            climb_rate_cmd,
            state.velocity[2],  # Current climb rate
            dt
        )
        pitch_cmd = np.clip(pitch_cmd, -np.radians(15), np.radians(15))
        
        # Return attitude command for Level 3
        return ControlCommand(
            mode=ControlMode.STICK_THROTTLE,
            roll_cmd=roll_cmd / np.radians(45),  # Normalize
            pitch_cmd=pitch_cmd / np.radians(30),  # Normalize
            yaw_cmd=0.0,
            throttle=throttle_cmd
        )
    
    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def reset(self):
        self.heading_pid.reset()
        self.speed_pid.reset()
        self.altitude_pid.reset()
```

## Level 1: Waypoint Navigation (Python)

**File**: `controllers/level1_waypoint.py`

```python
class Level1WaypointController:
    """Level 1: Waypoint to HSA commands."""
    
    def __init__(self, config):
        self.config = config
        self.guidance_type = config.get("guidance", "line_of_sight")
    
    def compute_command(
        self,
        state: AircraftState,
        input_command: ControlCommand,
        dt: float
    ) -> ControlCommand:
        """Convert waypoint to HSA commands."""
        
        waypoint = input_command.waypoint
        
        if self.guidance_type == "line_of_sight":
            hsa = self._line_of_sight_guidance(state, waypoint)
        elif self.guidance_type == "proportional_navigation":
            hsa = self._proportional_navigation(state, waypoint)
        else:
            raise ValueError(f"Unknown guidance: {self.guidance_type}")
        
        # Return HSA command for Level 2
        return ControlCommand(
            mode=ControlMode.HSA,
            heading=hsa["heading"],
            speed=hsa["speed"],
            altitude=hsa["altitude"]
        )
    
    def _line_of_sight_guidance(self, state, waypoint):
        """Simple line-of-sight guidance."""
        # Vector to waypoint
        delta_n = waypoint.north - state.north
        delta_e = waypoint.east - state.east
        
        # Desired heading
        heading_cmd = np.arctan2(delta_e, delta_n)
        
        # Desired altitude
        altitude_cmd = -waypoint.down
        
        # Speed command (use waypoint speed or maintain current)
        speed_cmd = waypoint.speed if waypoint.speed else state.airspeed
        
        return {
            "heading": heading_cmd,
            "speed": speed_cmd,
            "altitude": altitude_cmd
        }
    
    def _proportional_navigation(self, state, waypoint):
        """Proportional navigation guidance."""
        N = 3.0  # Navigation constant
        
        # Position error
        r = np.array([
            waypoint.north - state.north,
            waypoint.east - state.east,
            waypoint.down - state.down
        ])
        r_mag = np.linalg.norm(r)
        
        if r_mag < 0.1:
            # At waypoint, maintain current heading
            return {
                "heading": state.heading,
                "speed": state.airspeed,
                "altitude": -waypoint.down
            }
        
        # Line-of-sight vector
        los = r / r_mag
        
        # Velocity
        vel = np.array([
            state.velocity[0],  # North
            state.velocity[1],  # East
            state.velocity[2]   # Down
        ])
        
        # LOS rate
        los_rate = np.cross(r, vel) / (r_mag ** 2)
        
        # Commanded acceleration (perpendicular to LOS)
        accel_cmd = N * state.airspeed * los_rate
        
        # Convert to heading command
        heading_cmd = np.arctan2(los[1], los[0])
        
        return {
            "heading": heading_cmd,
            "speed": waypoint.speed if waypoint.speed else state.airspeed,
            "altitude": -waypoint.down
        }
    
    def reset(self):
        pass
```

## Control Level Coordinator

**File**: `controllers/flight_controller.py`

```python
class FlightController:
    """Coordinates all control levels."""
    
    def __init__(self, config):
        # Initialize all level controllers
        self.level1 = Level1WaypointController(config["level1"])
        self.level2 = Level2HSAController(config["level2"])
        self.level3 = Level3AttitudeController(config["level3"])
        self.level4_mixer = FixedWingMixer()  # C++
    
    def compute_control(
        self,
        state: AircraftState,
        command: ControlCommand,
        dt: float
    ) -> ControlSurfaces:
        """Process command through control hierarchy."""
        
        current_command = command
        
        # Execute control levels based on agent's level
        if command.mode == ControlMode.WAYPOINT:
            # Level 1 → 2 → 3 → 4
            current_command = self.level1.compute_command(state, current_command, dt)
            current_command = self.level2.compute_command(state, current_command, dt)
            current_command = self.level3.compute_command(state, current_command, dt)
        
        elif command.mode == ControlMode.HSA:
            # Skip Level 1, execute 2 → 3 → 4
            current_command = self.level2.compute_command(state, current_command, dt)
            current_command = self.level3.compute_command(state, current_command, dt)
        
        elif command.mode == ControlMode.STICK_THROTTLE:
            # Skip Levels 1-2, execute 3 → 4
            current_command = self.level3.compute_command(state, current_command, dt)
        
        elif command.mode == ControlMode.SURFACE:
            # Direct to mixer (Level 4 only)
            pass
        
        # Final: Mix to actuator commands
        surfaces = self.level4_mixer.mix({
            "roll": current_command.aileron,
            "pitch": current_command.elevator,
            "yaw": current_command.rudder,
            "throttle": current_command.throttle
        })
        
        return surfaces
    
    def reset(self):
        self.level1.reset()
        self.level2.reset()
        self.level3.reset()
```

## PID Tuning Guidelines

### Level 3 (Attitude) Tuning

**Process**:
1. Start with all gains at 0
2. Increase Kp until oscillation
3. Reduce Kp by 50%
4. Increase Kd to dampen oscillations
5. Add Ki slowly to remove steady-state error

**Typical Values** (fixed-wing):
```yaml
roll:
  kp: 0.15
  ki: 0.2
  kd: 0.0002
  
pitch:
  kp: 0.15
  ki: 0.2
  kd: 0.0002
  
yaw:
  kp: 0.3
  ki: 0.05
  kd: 0.00015
```

### Level 2 (HSA) Tuning

```yaml
heading:
  kp: 2.0
  ki: 0.0
  kd: 0.5
  
speed:
  kp: 0.1
  ki: 0.01
  kd: 0.0
  
altitude:
  kp: 0.05
  ki: 0.01
  kd: 0.2
```

---

**Document Status**: ✅ Complete
**Last Updated**: 2025-10-09
**Related Documents**:
- 03_CONTROL_HIERARCHY.md (control levels)
- dRehmFlight source code (reference)
