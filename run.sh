#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=20 -t=ds_test332 -lr 0.001 -bs 16 -lt=icnn_loss -a 1.0 -p 1.0 -q 1.0 -r 1.0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=20 -t=ds_test333 -lr 0.001 -bs 16 -lt=icnn_loss -a 1.0 -p 0.75 -q 0.75 -r 0.75
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=20 -t=ds_test334 -lr 0.001 -bs 16 -lt=icnn_loss -a 1.0 -p 0.5 -q 0.5 -r 0.5
