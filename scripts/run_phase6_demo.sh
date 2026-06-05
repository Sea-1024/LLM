#!/bin/bash
# Phase 6: Inference & Demo
# Launches the Gradio-based interactive demo application.
set -e

cd "$(dirname "$0")/.."

echo "=== Phase 6: Interactive Demo ==="
echo "Launching Gradio web interface..."
python -m src.phase6_inference.app

echo ""
echo "=== Phase 6 complete ==="
