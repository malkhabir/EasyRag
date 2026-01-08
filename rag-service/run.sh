#!/bin/bash
# Helper script to run commands with virtual environment
# Usage: ./run.sh <command>

set -e

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "Virtual environment created and dependencies installed!"
else
    source venv/bin/activate
fi

# Run the provided command
if [ $# -gt 0 ]; then
    echo "Running: $*"
    "$@"
else
    echo "Virtual environment activated. You can now run Python commands."
fi
