#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=20 -t=ds_test332 -lr 0.01 -bs 16 -lt=icnn_loss -a 1.0 -p 0.01 -q 0.01 -r 0.01
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=20 -t=ds_test333 -lr 0.01 -bs 16 -lt=icnn_loss -a 1.0 -p 0.001 -q 0.001 -r 0.001
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=20 -t=ds_test334 -lr 0.01 -bs 16 -lt=icnn_loss -a 1.0 -p 0.0001 -q 0.0001 -r 0.0001
