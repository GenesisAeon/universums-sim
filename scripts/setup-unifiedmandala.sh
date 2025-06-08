#!/bin/sh
# Setup script for UnifiedMandala environment
# Installs basic dependencies and prepares local environment

set -e

echo "Setting up UnifiedMandala..."

# Install Node and Python dependencies if package lists exist
if [ -f package.json ]; then
  npm install
fi
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi

echo "Setup complete"
