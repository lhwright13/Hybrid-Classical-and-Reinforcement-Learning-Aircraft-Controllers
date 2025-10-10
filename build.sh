#!/bin/bash
# Build script for aircraft controls C++ library and Python bindings

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building Aircraft Controls C++ Library ===${NC}"

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Cleaning existing build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Detect number of CPU cores for parallel build
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    NUM_CORES=$(sysctl -n hw.ncpu)
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    NUM_CORES=$(nproc)
else
    # Default to 4 cores
    NUM_CORES=4
fi

echo -e "${GREEN}Configuring with CMake...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_TESTS=OFF

echo -e "${GREEN}Building with $NUM_CORES cores...${NC}"
cmake --build . -j${NUM_CORES}

echo -e "${GREEN}Build complete!${NC}"
echo ""
echo -e "${GREEN}Python bindings available at:${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    ls -lh ../aircraft_controls_bindings*.so 2>/dev/null || ls -lh ../aircraft_controls_bindings*.dylib 2>/dev/null || echo -e "${YELLOW}No bindings found${NC}"
else
    ls -lh ../aircraft_controls_bindings*.so 2>/dev/null || echo -e "${YELLOW}No bindings found${NC}"
fi

echo ""
echo -e "${GREEN}To test the bindings, run:${NC}"
echo "  source venv/bin/activate"
echo "  python -c 'import aircraft_controls_bindings; print(aircraft_controls_bindings.__version__)'"
