#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=35 -t=ds_test347 -lr 0.01 -bs 16 -lt=icnn_loss -a 0.5 -p 0.8 -q 0.6 -r 0.1
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=35 -t=ds_test348 -lr 0.01 -bs 16 -lt=icnn_loss -a 0.5 -p 0.6 -q 0.8 -r 0.1
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=35 -t=ds_test349 -lr 0.01 -bs 16 -lt=icnn_loss -a 0.5 -p 0.8 -q 0.8 -r 0.1