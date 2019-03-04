#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=$1 python3 Classification.py $1 > classify_$1.txt