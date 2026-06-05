#!/bin/bash
# Phase 2: Tokenizer training and data tokenization
# Trains a BPE tokenizer and tokenizes the raw dataset.
set -e

cd "$(dirname "$0")/.."

echo "=== Phase 2: Tokenizer ==="
echo "Step 1/2: Training BPE tokenizer..."
python -m src.phase2_tokenizer.train_tokenizer

echo ""
echo "Step 2/2: Tokenizing datasets..."
python -m src.phase2_tokenizer.tokenize_data

echo ""
echo "=== Phase 2 complete ==="
