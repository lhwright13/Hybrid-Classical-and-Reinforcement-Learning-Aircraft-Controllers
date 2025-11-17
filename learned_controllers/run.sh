#!/bin/bash
# Helper script to run learned controller scripts with the correct venv

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/.."

# Python executable from venv
PYTHON="$PROJECT_ROOT/venv/bin/python"

# Check if venv exists
if [ ! -f "$PYTHON" ]; then
    echo "Error: Virtual environment not found at $PROJECT_ROOT/venv"
    echo "Please create a venv and install requirements:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Run the command
cd "$PROJECT_ROOT"
exec "$PYTHON" "$@"
