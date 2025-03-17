#!/bin/bash

# Check OS Type
if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
    # macOS/Linux activation
    source .venv/bin/activate
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows Git Bash activation
    source .venv/Scripts/activate
else
    echo "Unsupported OS. Please activate manually."
    exit 1
fi

echo "Virtual environment activated."
