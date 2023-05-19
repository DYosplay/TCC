#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=20 -t=ds_test341 -lr 0.01 -bs 16 -lt=icnn_loss -a 1.0 -p 0.4 -q 0.4 -r 0.4
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=20 -t=ds_test342 -lr 0.01 -bs 16 -lt=icnn_loss -a 1.0 -p 0.3 -q 0.3 -r 0.3
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=20 -t=ds_test343 -lr 0.01 -bs 16 -lt=icnn_loss -a 1.0 -p 0.6 -q 0.6 -r 0.6
