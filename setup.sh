#!/usr/bin/env bash
# Setup virtual environment and install dependencies for ADE Classifier & NRE Streamlit app.
set -e
cd "$(dirname "$0")"

echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install streamlit torch transformers pandas

echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Then run: ./run.sh"
