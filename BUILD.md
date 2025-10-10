# Build Instructions

This document explains how to build the C++ components and Python bindings for the aircraft controls project.

## Quick Start

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Configure and build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_PYTHON_BINDINGS=ON \
         -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
make -j$(sysctl -n hw.ncpu)  # macOS
# make -j$(nproc)  # Linux

# 3. Test bindings
cd ..
python -c "import aircraft_controls_bindings; print(aircraft_controls_bindings.__version__)"
```

## Prerequisites

### Required

- **C++ Compiler**: C++17 compatible
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Linux: GCC 7+ or Clang 5+

- **CMake**: Version 3.15 or higher
  ```bash
  # macOS
  brew install cmake

  # Linux
  sudo apt-get install cmake
  ```

- **Python**: 3.8 or higher (3.13.6 currently used)

- **pybind11**: Installed in virtual environment
  ```bash
  pip install pybind11
  ```

### Optional

- **Ninja**: Faster build system (optional but recommended)
  ```bash
  brew install ninja  # macOS
  sudo apt-get install ninja-build  # Linux
  ```

## Build Options

### CMake Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Release` | Build type: `Debug`, `Release`, `RelWithDebInfo` |
| `BUILD_PYTHON_BINDINGS` | `ON` | Build Python bindings with pybind11 |
| `BUILD_TESTS` | `ON` | Build C++ unit tests (currently not implemented) |
| `BUILD_SHARED_LIBS` | `ON` | Build shared libraries (.dylib/.so) vs static (.a) |

### Example Configurations

**Debug Build:**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
```

**Release Build with Optimizations:**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_FLAGS="-O3 -march=native" \
         -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
```

**Without Python Bindings (C++ only):**
```bash
cmake .. -DBUILD_PYTHON_BINDINGS=OFF
```

**Using Ninja:**
```bash
cmake .. -GNinja \
         -DCMAKE_BUILD_TYPE=Release \
         -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
ninja
```

## Build Process Details

### Step-by-Step Build

1. **Create build directory**
   ```bash
   mkdir -p build
   cd build
   ```

2. **Configure with CMake**
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_PYTHON_BINDINGS=ON \
            -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
   ```

3. **Build**
   ```bash
   # Use all CPU cores
   make -j$(sysctl -n hw.ncpu)  # macOS
   make -j$(nproc)              # Linux

   # Or specify cores manually
   make -j8
   ```

4. **Verify build**
   ```bash
   ls -lh ../aircraft_controls_bindings*.so  # Linux
   ls -lh ../aircraft_controls_bindings*.dylib  # macOS (but shows as .so in Python)
   ```

### What Gets Built

- **Core C++ Library**: `libaircraft_controls_core.dylib` (or `.so` on Linux)
  - Location: `build/libaircraft_controls_core.dylib`
  - Contains: PID controller and other C++ components

- **Python Bindings**: `aircraft_controls_bindings.cpython-313-darwin.so`
  - Location: Project root (automatically copied)
  - Contains: Python-accessible wrappers for C++ classes

## Testing

### Test Python Bindings

```bash
source venv/bin/activate

# Quick test
python -c "import aircraft_controls_bindings as acb; print(acb.__version__)"

# Run full test suite
pytest tests/test_pid_bindings.py -v

# Test PID controller
python << EOF
import aircraft_controls_bindings as acb

config = acb.PIDConfig()
config.gains = acb.PIDGains(1.0, 0.1, 0.05)
pid = acb.PIDController(config)

output = pid.compute(setpoint=100.0, measurement=95.0, dt=0.01)
print(f"PID Output: {output}")
print(f"Error: {pid.get_error()}")
EOF
```

### Run All Tests

```bash
# All Python tests (interfaces + PID bindings)
pytest tests/ -v

# With coverage
pytest tests/ --cov=interfaces --cov=controllers -v
```

## Troubleshooting

### Issue: CMake can't find pybind11

**Error:**
```
Could not find a package configuration file provided by "pybind11"
```

**Solution:**
```bash
# Activate venv first
source venv/bin/activate

# Install pybind11
pip install pybind11

# Build with explicit path
cmake .. -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
```

### Issue: Wrong Python version

**Error:**
```
Found Python3: /usr/bin/python3 (found version "3.9.0")
```

**Solution:**
```bash
# Specify Python executable explicitly
cmake .. -DPython3_EXECUTABLE=$(which python3)
```

### Issue: Bindings not found when importing

**Error:**
```python
>>> import aircraft_controls_bindings
ModuleNotFoundError: No module named 'aircraft_controls_bindings'
```

**Solution:**
```bash
# Make sure you're in project root
cd /path/to/controls

# Check bindings file exists
ls -lh aircraft_controls_bindings*.so

# Make sure venv is activated
source venv/bin/activate

# Rebuild if necessary
cd build && make && cd ..
```

### Issue: Build fails with C++ errors

**Common causes:**
1. C++ compiler too old (need C++17)
2. Missing headers

**Solution:**
```bash
# Check compiler version
c++ --version  # Should be Clang 5+ or GCC 7+

# Update compiler
# macOS:
xcode-select --install

# Linux:
sudo apt-get update
sudo apt-get install build-essential
```

### Issue: Linker errors on macOS

**Error:**
```
ld: library not found for -lc++
```

**Solution:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# If that doesn't work, install full Xcode from App Store
```

## Clean Build

To start fresh:

```bash
# Remove build directory
rm -rf build/

# Remove built bindings
rm -f aircraft_controls_bindings*.so

# Rebuild from scratch
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
make -j8
```

## Advanced

### Cross-Compilation

For embedded targets (e.g., Raspberry Pi):

```bash
cmake .. -DCMAKE_TOOLCHAIN_FILE=/path/to/toolchain.cmake \
         -DCMAKE_BUILD_TYPE=Release
```

### Static Analysis

```bash
# With clang-tidy
cmake .. -DCMAKE_CXX_CLANG_TIDY="clang-tidy;-checks=*"
make
```

### Install to System

```bash
# Install to /usr/local
sudo make install

# Install to custom prefix
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local
make install
```

## Performance Optimization

### Compiler Flags

For maximum performance on specific hardware:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -flto"
```

### Link-Time Optimization (LTO)

```bash
cmake .. -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
```

## CI/CD

For automated builds:

```bash
# Minimal CI build
cmake -B build -S . \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_PYTHON_BINDINGS=ON
cmake --build build -j$(nproc)
```

## Project Structure

After building, your project should look like:

```
controls/
├── build/                              # Build artifacts
│   ├── CMakeFiles/
│   ├── Makefile
│   ├── libaircraft_controls_core.dylib
│   └── ...
├── cpp/
│   ├── include/
│   │   └── pid_controller.h           # C++ headers
│   ├── src/
│   │   └── pid_controller.cpp         # C++ source
│   └── bindings/
│       └── bindings.cpp               # Pybind11 bindings
├── aircraft_controls_bindings.*.so    # Python module (auto-copied)
├── CMakeLists.txt                     # Build configuration
└── ...
```

## Next Steps

After successful build:

1. Run tests: `pytest tests/ -v`
2. Read integration guide: See `design_docs/05_AGENT_INTERFACE_CONTROL.md`
3. Start Phase 2: Simulation backend implementation

---

**Build System**: CMake 3.15+
**Last Updated**: 2025-10-10
**Platform Tested**: macOS (darwin), Linux expected to work
