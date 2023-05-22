#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=35 -t=ds_test353 -lr 0.01 -bs 16 -lt=icnn_loss -a 0.5 -p 8 -q 8 -r 8
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=35 -t=ds_test354 -lr 0.01 -bs 16 -lt=icnn_loss -a 0.5 -p 10 -q 10 -r 10
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=35 -t=ds_test355 -lr 0.01 -bs 16 -lt=icnn_loss -a 0.5 -p 12 -q 12 -r 12



