#!/usr/bin/env bash
set -e

# Optional: export GEMINI_API_KEY if already present in the environment
if [ -n "${GEMINI_API_KEY:-}" ]; then
  export GEMINI_API_KEY="$GEMINI_API_KEY"
fi

exec streamlit run src/ui/streamlit_app.py


