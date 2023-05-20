#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=35 -t=ds_test350 -lr 0.01 -bs 16 -lt=icnn_loss -a 0.5 -p 2 -q 2 -r 2
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=35 -t=ds_test351 -lr 0.01 -bs 16 -lt=icnn_loss -a 0.5 -p 4 -q 4 -r 4
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=35 -t=ds_test352 -lr 0.01 -bs 16 -lt=icnn_loss -a 0.5 -p 6 -q 6 -r 6
