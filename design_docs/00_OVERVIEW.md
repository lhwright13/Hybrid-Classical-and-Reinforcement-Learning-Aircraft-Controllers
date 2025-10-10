# System Overview - Multi-Level Aircraft Control & RL Training Platform

## Project Vision

This project creates a **multi-level aircraft control system** that enables:

1. **Research and demonstration** of classical and RL-based flight control algorithms
2. **Training RL agents** at multiple levels of control abstraction (waypoint → surface deflection)
3. **Seamless sim-to-real transfer** where agents trained in simulation deploy to real hardware
4. **Complete swappability** of agents, aircraft backends, and sensors

## Core Philosophy

### Abstraction Through Control Levels

Aircraft control naturally forms a hierarchy of abstraction:

```
Level 1: Waypoint Navigation    ← Highest abstraction, hardest to learn
Level 2: HSA (Heading/Speed/Alt) ← High-level flight states
Level 3: Stick & Throttle        ← Medium-level attitude control
Level 4: Control Surfaces        ← Lowest abstraction, easiest to learn
```

**Key Insight**: Agents can command at ANY level. Training an agent at Level 4 (surfaces) is fundamentally different from Level 1 (waypoints):
- **Level 4**: Dense rewards, short horizon, sample efficient, direct control
- **Level 1**: Sparse rewards, long horizon, sample inefficient, requires navigation

### Multi-Level Agent Training

Agents are not restricted to one level:

- **Single-level agents**: Trained at one fixed level (e.g., Level 3 stick commands)
- **Hierarchical agents**: High-level agent (Level 1) commands low-level agent (Level 3)
- **Adaptive agents**: Dynamically switch levels based on task demands
- **Hybrid agents**: Combine RL at one level with classical control at another

### Sim-to-Real Through Abstraction

Every component has an abstract interface:

```
Agent → [Aircraft Interface] → {Simulation | Real Hardware}
Agent → [Sensor Interface]   → {Perfect Sim | Noisy Sim | Real Sensors}
```

**Zero code changes** to switch backends—only configuration changes.

## Use Cases

### Use Case 1: Algorithm Comparison
- Compare classical PID vs RL agents at each control level
- Benchmark: Which level is best for learning acrobatic maneuvers?
- Metric: Sample efficiency, performance, robustness

### Use Case 2: RL Research
- Train agents at all 4 levels
- Study transfer learning between levels
- Experiment with hierarchical RL
- Test meta-learning for level selection

### Use Case 3: Real-World Deployment
- Train agent in simulation (JSBSim)
- Validate in Hardware-in-the-Loop (HIL)
- Deploy to real aircraft (Teensy/dRehmFlight hardware)
- Monitor performance and collect real-world data

### Use Case 4: Novel Aircraft Design
- Test unconventional aircraft (VTOL, cyclocopter, tailsitter)
- Use Level 4 agents for direct surface control
- Discover non-intuitive control strategies

### Use Case 5: Formation Flight & Swarms
- Multiple agents commanding at Level 1 (waypoints) or Level 2 (HSA)
- Decentralized multi-agent RL
- Coordination and collision avoidance

## Key Features

### 1. Multi-Level Control Architecture
- 4 distinct control levels with well-defined interfaces
- Classical controllers at each level (from dRehmFlight)
- RL agents can replace or augment any level
- Hierarchical composition of levels

### 2. Hybrid Python/C++ Implementation
- **Python**: High-level logic, RL training, interfaces, visualization
- **C++**: Performance-critical PID controllers (100-500 Hz)
- **Pybind11**: Seamless Python ↔ C++ integration

### 3. Simulation & Hardware Backends
- **JSBSim**: High-fidelity aerodynamic simulation
- **Simplified 6-DOF**: Fast simulation for rapid prototyping
- **Hardware Interface**: Direct integration with Teensy/dRehmFlight
- **HIL Support**: Hardware-in-the-loop testing

### 4. RL Training Infrastructure
- **Multiple frameworks**: Stable-Baselines3, RLlib, CleanRL
- **Vectorized environments**: Parallel training across many sims
- **Curriculum learning**: Progressive difficulty
- **Domain randomization**: Robustness to sim-to-real gap
- **Level-specific training**: Optimized for each control level

### 5. Visualization & Monitoring
- **Real-time dashboard**: Plotly Dash web interface
- **3D visualization**: Live aircraft orientation and trajectory
- **Telemetry plots**: All state variables and control outputs
- **Training monitoring**: RL learning curves, performance metrics

### 6. Complete Abstraction
- **Agent Interface**: Any agent type (classical, RL, hybrid, human)
- **Aircraft Interface**: Any backend (sim, real, HIL)
- **Sensor Interface**: Any sensor source (perfect, noisy, real)
- **Swappability**: Change any component via configuration

## Design Principles

### Principle 1: Interfaces Over Implementations
Every component is defined by its interface contract first. Implementations are swappable as long as they satisfy the interface.

### Principle 2: Configuration Over Code
Switching agents, backends, or sensors should require only configuration changes, not code modifications.

### Principle 3: Fail-Safe Layering
Higher-level control failures can fall back to lower levels. Emergency override always available.

### Principle 4: Observable Everything
Every state transition, control command, and performance metric is logged for analysis.

### Principle 5: Reproducible Research
All experiments are fully reproducible with seeds, versioned configs, and deterministic simulation.

## System Boundaries

### In Scope
- Fixed-wing aircraft control
- Quadrotor/multirotor control
- VTOL aircraft
- Single aircraft (primary) with multi-agent extensions
- Simulation and real hardware deployment
- Classical and RL control algorithms

### Out of Scope (Future Work)
- Helicopters (complex rotor dynamics)
- Large commercial aircraft
- Space vehicles (orbital mechanics)
- Full autopilot system (landing, taxi, etc.)
- Certified safety-critical systems

## Target Users

### Researcher
Experiments with new RL algorithms at different control levels. Needs flexibility, logging, and reproducibility.

### Hobbyist
Wants to fly custom aircraft with autonomous capabilities. Needs working examples and easy integration.

### Student
Learning about flight control and RL. Needs clear documentation and educational demos.

### Developer
Integrating new agents or aircraft models. Needs clear interfaces and integration guides.

## Success Metrics

The project is successful if:

1. ✅ **An RL agent can be trained at ANY control level (1-4) and deployed to real hardware**
2. ✅ **Switching from simulation to hardware requires ONLY config changes (zero code changes)**
3. ✅ **A new agent can be integrated in < 1 day using the integration guide**
4. ✅ **Hierarchical agents can be composed from single-level agents**
5. ✅ **Performance baselines exist for classical controllers at all levels**
6. ✅ **All experiments are reproducible from config files**

## Document Reading Guide

### For Understanding the System
1. Start here (00_OVERVIEW.md)
2. Read 01_ARCHITECTURE.md for system-wide view
3. Read 03_CONTROL_HIERARCHY.md for control level details

### For Implementing Agents
1. Read 05_AGENT_INTERFACE_CONTROL.md (integration guide)
2. Read 02_ABSTRACTION_LAYERS.md (interface contracts)
3. Read 06_RL_AGENT_TRAINING.md (if training RL agents)

### For Hardware Integration
1. Read 08_HARDWARE_INTERFACE.md
2. Read 11_DEPLOYMENT.md (sim-to-real workflow)
3. Read 07_SIMULATION_INTERFACE.md (for comparison)

### For Implementation
1. Read 12_IMPLEMENTATION_ROADMAP.md first
2. Follow dependency order
3. Reference other docs as needed

## Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| RL Training | Stable-Baselines3, RLlib | Agent training |
| Simulation | JSBSim, Custom 6-DOF | Physics simulation |
| Core Control | C++17, Pybind11 | Fast PID controllers |
| High-Level Logic | Python 3.8+ | Agents, interfaces, modes |
| Visualization | Plotly Dash, Matplotlib | Real-time monitoring |
| Data | Pandas, HDF5 | Logging and analysis |
| Build | CMake, setuptools | Compilation |
| Testing | pytest | Validation |

## Next Steps

After reading this overview:
1. Proceed to **01_ARCHITECTURE.md** for detailed system design
2. Then **03_CONTROL_HIERARCHY.md** to understand multi-level control
3. Then **05_AGENT_INTERFACE_CONTROL.md** to integrate your first agent

---

**Document Status**: ✅ Complete
**Last Updated**: 2025-10-09
**Related Documents**: All (entry point)
