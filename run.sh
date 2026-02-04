#!/usr/bin/env bash
# Run the Streamlit app (ADE Classifier & Relational Extraction).
set -e
cd "$(dirname "$0")"

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

exec streamlit run app.py --server.port 8501
