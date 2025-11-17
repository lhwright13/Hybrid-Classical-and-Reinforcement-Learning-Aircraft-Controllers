# Contributing to Multi-Level Flight Control

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Style Guidelines](#code-style-guidelines)
4. [Testing Requirements](#testing-requirements)
5. [Pull Request Process](#pull-request-process)
6. [Areas for Contribution](#areas-for-contribution)
7. [Research Collaboration](#research-collaboration)

---

## Getting Started

### Before You Start

- Read the [README.md](README.md) to understand the project
- Check existing [Issues](https://github.com/yourusername/controls/issues) and [Pull Requests](https://github.com/yourusername/controls/pulls)
- Review the [ROADMAP.md](ROADMAP.md) to see planned features
- Join discussions in [GitHub Discussions](https://github.com/yourusername/controls/discussions)

### Ways to Contribute

- 🐛 **Bug Reports**: Found a bug? Open an issue with reproduction steps
- 💡 **Feature Requests**: Have an idea? Propose it in Discussions
- 📝 **Documentation**: Improve docs, add examples, fix typos
- 🧪 **Tests**: Add test coverage, improve test quality
- 🚀 **Code**: Implement features, fix bugs, optimize performance
- 🔬 **Research**: Contribute experiments, benchmarks, analysis

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/controls.git
cd controls

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_USERNAME/controls.git
```

### 2. Create Development Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Build C++ components
./build.sh
```

### 3. Verify Setup

```bash
# Run tests to ensure everything works
pytest tests/ -v

# Run a simple example
python examples/01_hello_controls.py
```

### 4. Create Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

---

## Code Style Guidelines

### Python Code Style

We follow **PEP 8** with some modifications.

#### Formatting

Use **Black** for automatic formatting:

```bash
# Format code
black controllers/ interfaces/ simulation/ gui/ examples/

# Check formatting
black --check controllers/
```

**Black Configuration** (`.black.toml`):
```toml
[tool.black]
line-length = 100
target-version = ['py38']
```

#### Linting

Use **Flake8** for linting:

```bash
# Run linter
flake8 controllers/ interfaces/ simulation/

# With specific rules
flake8 --max-line-length=100 --ignore=E203,W503 controllers/
```

#### Type Checking

Use **mypy** for static type checking:

```bash
# Type check
mypy controllers/ interfaces/ simulation/

# With strict mode
mypy --strict controllers/my_module.py
```

### C++ Code Style

- Follow **C++17** standards
- Use **clang-format** with Google style
- RAII for resource management
- `const` correctness
- Smart pointers over raw pointers

```bash
# Format C++ code
clang-format -i core/**/*.cpp core/**/*.hpp
```

### Documentation Style

#### Docstrings

Use **Google-style docstrings** for Python:

```python
def compute_action(
    self,
    command: ControlCommand,
    state: AircraftState
) -> ControlSurfaces:
    """Compute control action from command and state.

    This function implements the core control logic, mapping from
    high-level commands to low-level surface deflections.

    Args:
        command: Control command specifying desired behavior
        state: Current aircraft state

    Returns:
        ControlSurfaces with aileron, elevator, rudder, throttle

    Raises:
        ValueError: If command mode doesn't match agent level
        AssertionError: If state is invalid

    Example:
        >>> agent = RateAgent(config)
        >>> command = ControlCommand(mode=ControlMode.RATE, roll_rate=0.5)
        >>> surfaces = agent.compute_action(command, state)
    """
```

#### Comments

- **Why** over **what**: Explain reasoning, not obvious code
- Use `# TODO:` for future work
- Use `# FIXME:` for known issues
- Use `# NOTE:` for important clarifications

```python
# Good: Explains why
# Use RK4 integration for better stability at large timesteps
state = rk4_step(state, dt)

# Bad: States the obvious
# Call rk4_step function
state = rk4_step(state, dt)
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `RateAgent`, `AircraftState`)
- **Functions/Methods**: `snake_case` (e.g., `compute_action`, `get_state`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_ROLL_RATE`, `DEFAULT_DT`)
- **Private**: Leading underscore (e.g., `_internal_state`, `_validate_input`)
- **Type Variables**: Single capital letter (e.g., `T`, `StateT`)

### File Organization

```python
"""Module docstring explaining purpose.

Longer description if needed.
"""

# Standard library imports
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from controllers import RateAgent
from simulation import SimulationAircraftBackend

# Constants
DEFAULT_DT = 0.01
MAX_ITERATIONS = 1000

# Module-level code
```

---

## Testing Requirements

### Test Coverage

- **Minimum**: 80% code coverage for new code
- **Target**: 90%+ for core modules (controllers, simulation, interfaces)
- Use `pytest-cov` to measure coverage

```bash
# Run with coverage
pytest --cov=controllers --cov=interfaces --cov=simulation tests/

# Generate HTML report
pytest --cov=controllers --cov-report=html tests/
open htmlcov/index.html
```

### Writing Tests

#### Test Structure

```python
"""Test module for RateAgent."""

import pytest
import numpy as np
from controllers import RateAgent, ControlCommand, ControlMode

@pytest.fixture
def config():
    """Create test configuration."""
    # Setup
    config = create_test_config()
    yield config
    # Teardown (if needed)

class TestRateAgent:
    """Tests for RateAgent."""

    def test_initialization(self, config):
        """Test agent initializes correctly."""
        agent = RateAgent(config)
        assert agent.get_control_level() == ControlMode.RATE

    def test_compute_action_valid_input(self, config):
        """Test compute_action with valid input."""
        agent = RateAgent(config)
        command = ControlCommand(mode=ControlMode.RATE, roll_rate=0.5)
        state = create_test_state()

        surfaces = agent.compute_action(command, state)

        assert -1.0 <= surfaces.aileron <= 1.0
        assert surfaces is not None

    def test_compute_action_invalid_mode(self, config):
        """Test compute_action raises on invalid mode."""
        agent = RateAgent(config)
        command = ControlCommand(mode=ControlMode.ATTITUDE, roll_angle=0.5)
        state = create_test_state()

        with pytest.raises(AssertionError):
            agent.compute_action(command, state)

    @pytest.mark.parametrize("roll_rate", [0.0, 0.5, 1.0, -0.5])
    def test_various_roll_rates(self, config, roll_rate):
        """Test agent handles various roll rates."""
        agent = RateAgent(config)
        command = ControlCommand(mode=ControlMode.RATE, roll_rate=roll_rate)
        state = create_test_state()

        surfaces = agent.compute_action(command, state)
        assert surfaces is not None
```

#### Test Categories

- **Unit Tests**: Test individual functions/methods
- **Integration Tests**: Test component interactions
- **System Tests**: Test full workflows
- **Regression Tests**: Ensure bugs stay fixed

#### Test Markers

```python
@pytest.mark.slow  # For tests >1 second
@pytest.mark.integration  # Integration tests
@pytest.mark.gpu  # Requires GPU
@pytest.mark.hardware  # Requires hardware
```

Run specific markers:
```bash
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Only integration tests
```

### Running Tests

```bash
# All tests
pytest tests/

# Specific file
pytest tests/test_rate_agent.py

# Specific test
pytest tests/test_rate_agent.py::TestRateAgent::test_initialization

# Verbose mode
pytest tests/ -v

# Stop on first failure
pytest tests/ -x

# Run in parallel (requires pytest-xdist)
pytest tests/ -n auto
```

---

## Pull Request Process

### 1. Ensure Quality

Before submitting:

```bash
# Format code
black controllers/ interfaces/ simulation/

# Lint code
flake8 controllers/ interfaces/ simulation/

# Type check
mypy controllers/ interfaces/ simulation/

# Run tests
pytest tests/ -v

# Check coverage
pytest --cov=controllers --cov=interfaces tests/
```

### 2. Update Documentation

- Add docstrings to new functions/classes
- Update relevant README sections
- Add example usage if applicable
- Update CHANGELOG.md

### 3. Commit Guidelines

Follow **Conventional Commits** format:

```
type(scope): brief description

Longer description if needed.

Fixes #123
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Add/update tests
- `perf`: Performance improvement
- `chore`: Maintenance tasks

**Examples**:
```bash
git commit -m "feat(controllers): add learned rate agent"
git commit -m "fix(simulation): correct roll angle calculation"
git commit -m "docs(readme): update installation instructions"
git commit -m "test(rate_agent): add edge case tests"
```

### 4. Push and Create PR

```bash
# Push branch
git push origin feature/your-feature-name

# Create PR on GitHub
# Use the PR template, provide context, link issues
```

### 5. PR Review Process

- Maintainers will review your PR
- Address feedback in new commits
- Once approved, maintainers will merge
- PR may be squashed or rebased before merge

### PR Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Added tests for new code
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (for features/fixes)
- [ ] Commit messages follow Conventional Commits
- [ ] PR description explains changes clearly

---

## Areas for Contribution

### High Priority

1. **RL Training** (Phase 5)
   - Train controllers for Levels 1-3
   - Hyperparameter optimization
   - Curriculum learning improvements
   - Ablation studies

2. **PID Tuning** (Phase 3)
   - Implement auto-tuning algorithms
   - Ziegler-Nichols method
   - Genetic algorithm tuning
   - Systematic gain sweep tools

3. **Web Dashboard** (Phase 4)
   - Plotly Dash interface
   - Real-time telemetry streaming
   - Multi-aircraft visualization
   - Configuration management UI

4. **Documentation**
   - Tutorial videos
   - More examples
   - API reference
   - Architecture deep-dives

### Medium Priority

5. **Validation** (Phase 2/3)
   - Additional JSBSim test scenarios
   - Wind disturbance testing
   - Monte Carlo validation
   - Comparison with real flight data

6. **Performance**
   - Simulation speed optimization
   - GPU acceleration for RL training
   - Vectorized environments
   - Profiling and bottleneck analysis

7. **Visualization**
   - Better 3D models
   - Flight path prediction
   - Augmented reality overlay
   - VR flight interface

### Future Work

8. **Hardware** (Phase 6)
   - Teensy integration
   - MAVLink protocol
   - HIL testing framework
   - Safety systems

9. **Advanced Agents** (Phase 7)
   - Hierarchical RL
   - Multi-agent coordination
   - Adaptive switching
   - Hybrid architectures

### Good First Issues

New contributors should look for issues labeled:
- `good first issue`
- `documentation`
- `help wanted`
- `beginner-friendly`

Example good first contributions:
- Fix typos in documentation
- Add unit tests for uncovered code
- Implement simple examples
- Improve error messages

---

## Research Collaboration

### Academic Partnerships

We welcome collaboration with researchers! Areas of interest:

1. **Multi-Level Reinforcement Learning**
   - Sample efficiency across abstraction levels
   - Transfer learning between levels
   - Hierarchical policy learning

2. **Sim-to-Real Transfer**
   - Domain randomization strategies
   - Reality gap analysis
   - Adaptation techniques

3. **Safety and Verification**
   - Formal verification of learned controllers
   - Safety shields and fallback policies
   - Certified robustness

4. **Human-Robot Interaction**
   - Shared autonomy
   - Adjustable autonomy
   - Trust and transparency

### Publication Policy

- Code contributors may be co-authors on papers using this work
- Please cite the repository in publications
- Coordinate with maintainers before submitting papers

### Research Data Sharing

- Training logs: Share via TensorBoard.dev or Weights & Biases
- Datasets: Use Hugging Face Datasets or Zenodo
- Models: Upload to Hugging Face Hub
- Results: Include in `learned_controllers/results/`

---

## Code of Conduct

### Our Standards

- **Respectful**: Treat everyone with respect and kindness
- **Collaborative**: Work together, help each other learn
- **Constructive**: Provide actionable, helpful feedback
- **Inclusive**: Welcome contributors from all backgrounds

### Unacceptable Behavior

- Harassment, discrimination, or hate speech
- Personal attacks or insults
- Trolling or inflammatory comments
- Sharing private information without consent

### Enforcement

Violations will result in:
1. Warning
2. Temporary ban
3. Permanent ban

Report issues to: [maintainer email]

---

## Questions?

- **GitHub Discussions**: For general questions and ideas
- **Issues**: For bugs and feature requests
- **Email**: For private inquiries or collaboration

---

Thank you for contributing to Multi-Level Flight Control! 🚁

Every contribution, no matter how small, helps advance the state of intelligent flight control systems.
