#!/bin/bash -x

python -m pytest --pyargs cu_cat --cov=cu_cat -n auto
