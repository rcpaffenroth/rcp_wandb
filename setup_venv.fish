echo "You need to source this script using . ./setup_venv.fish"

# Install a local Python virtual environment

# This is the fish version
if test -d venv
    echo "Removing pre-exising venv"
    rm -rf venv
end

# Create a virtual environment
python3 -m venv venv

# Active the environment so this is my default python
. ./venv/bin/activate.fish

poetry install --with dev
