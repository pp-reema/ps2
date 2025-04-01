#!/usr/bin/env bash
# Install the correct Python version
pyenv install 3.12.9
pyenv global 3.12.9

# Install system dependencies
apt-get update && apt-get install -y portaudio19-dev

# Install Python dependencies
pip install -r requirements.txt
