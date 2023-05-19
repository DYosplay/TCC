#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=20 -t=ds_test335 -lr 0.01 -bs 16 -lt=icnn_loss -a 1.0 -p 0.25 -q 0.25 -r 0.25
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=20 -t=ds_test336 -lr 0.01 -bs 16 -lt=icnn_loss -a 1.0 -p 0.1 -q 0.1 -r 0.1
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=20 -t=ds_test337 -lr 0.01 -bs 16 -lt=icnn_loss -a 1.0 -p 0.3 -q 0.3 -r 0.3
