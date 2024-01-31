#!/bin/bash
set -ex

# Run from project root
# - Args get passed to pytest phase
# Non-zero exit code on fail

# Assume minimal env (pandas) and working gapencoder

python -m pytest --version

python -B -m pytest -vv \
    cu_cat/tests/test_table_vectorizer.py 
