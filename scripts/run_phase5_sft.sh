#!/bin/bash
# Phase 5: Supervised Fine-Tuning (SFT)
# Prepares SFT data, fine-tunes the pretrained model, and evaluates.
set -e

cd "$(dirname "$0")/.."

echo "=== Phase 5: Supervised Fine-Tuning ==="
echo "Step 1/3: Preparing SFT data..."
python -m src.phase5_sft.data_prepare

echo ""
echo "Step 2/3: Running SFT training..."
python -m src.phase5_sft.trainer

echo ""
echo "Step 3/3: Evaluating fine-tuned model..."
python -m src.phase5_sft.evaluate

echo ""
echo "=== Phase 5 complete ==="
