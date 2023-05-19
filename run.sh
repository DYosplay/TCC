#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=20 -t=ds_test332 -lr 0.01 -bs 16 -lt=icnn_loss -a 1.0 -p 0.5 -q 0.3 -r 0.25
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=20 -t=ds_test333 -lr 0.01 -bs 16 -lt=icnn_loss -a 1.0 -p 0.7 -q 0.5 -r 0.3
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=20 -t=ds_test334 -lr 0.01 -bs 16 -lt=icnn_loss -a 1.0 -p 0.4 -q 0.25 -r 0.5
