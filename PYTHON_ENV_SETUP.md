# Python Virtual Environment Setup

This guide explains how to set up and use the Python virtual environment for the controls project.

## Quick Start

```bash
# Activate the virtual environment
source venv/bin/activate

# Run tests
pytest tests/

# Deactivate when done
deactivate
```

## Initial Setup (Already Done)

The virtual environment has already been created and configured. If you need to recreate it:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

## Using the Virtual Environment

### Activate

Before working on the project, always activate the virtual environment:

```bash
source venv/bin/activate
```

You'll see `(venv)` prepended to your terminal prompt when active.

### Deactivate

When you're done working:

```bash
deactivate
```

## Running Tests

With the virtual environment activated:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_interfaces.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=interfaces --cov=controllers

# Run specific test class or function
pytest tests/test_interfaces.py::TestBaseAgent
pytest tests/test_interfaces.py::TestBaseAgent::test_cannot_instantiate_abstract_class
```

## Installed Packages

Key packages installed in this environment:

**Core Dependencies:**
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `pyproj` - Geodetic coordinate transformations

**Simulation & Visualization:**
- `matplotlib` - 2D plotting
- `plotly` - Interactive plots
- `pyvista` - 3D visualization
- `dash` - Web-based dashboards

**Data Management:**
- `pandas` - Data analysis
- `h5py` - HDF5 file format

**Hardware Interface:**
- `pyserial` - Serial port communication

**C++ Integration:**
- `pybind11` - Python/C++ bindings

**Testing & Development:**
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `black` - Code formatting
- `flake8` - Linting
- `mypy` - Type checking

See `requirements.txt` for the complete list with version constraints.

## Adding New Dependencies

If you need to add a new package:

```bash
# Activate venv first
source venv/bin/activate

# Install package
pip install package-name

# Update requirements.txt
pip freeze > requirements.txt
```

Or manually add to `requirements.txt` with a version constraint:
```
package-name>=1.2.3
```

Then install:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### Virtual Environment Not Found

If `venv/` directory is missing, recreate it:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Wrong Python Version

Check your Python version:

```bash
python3 --version
```

This project requires Python 3.8+. The current environment uses Python 3.13.6.

### Import Errors

Make sure:
1. Virtual environment is activated
2. You're in the project root directory
3. All packages are installed: `pip install -r requirements.txt`

### Permission Errors

If you get permission errors during installation, make sure you're installing in the virtual environment (not system-wide).

## IDE Integration

### VSCode

VSCode should automatically detect the virtual environment. If not:

1. Open Command Palette (`Cmd+Shift+P`)
2. Select "Python: Select Interpreter"
3. Choose `./venv/bin/python`

### PyCharm

1. Go to Settings → Project → Python Interpreter
2. Click gear icon → Add
3. Select "Existing environment"
4. Navigate to `venv/bin/python`

## CI/CD Considerations

For CI/CD pipelines, use Docker instead (see `README_DOCKER.md`):

```bash
docker-compose up dev
```

The Docker environment includes all dependencies and doesn't require virtual environment setup.

## Notes

- The `venv/` directory is in `.gitignore` and should never be committed
- Virtual environments are machine-specific and should be recreated on each machine
- For production deployment on hardware, consider using Docker or system packages
- The virtual environment is only needed for development; Docker handles deployment

---

**Environment Info:**
- Python Version: 3.13.6
- Platform: macOS (darwin)
- Created: 2025-10-10
