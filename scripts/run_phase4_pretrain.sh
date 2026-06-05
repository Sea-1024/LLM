#!/bin/bash
# Phase 4: Pretraining
# Trains the MiniLLM model from scratch on the tokenized corpus.
set -e

cd "$(dirname "$0")/.."

echo "=== Phase 4: Pretraining ==="
python -m src.phase4_pretrain.trainer

echo ""
echo "=== Phase 4 complete ==="
