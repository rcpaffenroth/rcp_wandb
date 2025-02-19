#! /bin/bash

echo "You need to source this script using . ./setup_venv.sh"

# Install a local Python virtual environment
if [ -d "venv" ]; then
  echo "Removing pre-exising venv";
  rm -rf venv;
fi

# Create a virtual environment
python3 -m venv venv;

# Active the environment so this is my default python
. ./venv/bin/activate

pip install --upgrade pip
pip install --editable .
