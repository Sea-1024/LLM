#!/bin/bash
# Phase 1: Data preparation
# Downloads raw datasets and preprocesses them into tokenized binary format.
set -e

cd "$(dirname "$0")/.."

echo "=== Phase 1: Data Preparation ==="
echo "Step 1/2: Downloading datasets..."
python -m src.phase1_data.download

echo ""
echo "Step 2/2: Preprocessing / tokenizing data..."
python -m src.phase1_data.preprocess

echo ""
echo "=== Phase 1 complete ==="
