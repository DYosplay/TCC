#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=30 -t=ds_test311 -tl=0.9 -lr 0.01 -bs 16
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=30 -t=ds_test312 -tl=0.8 -lr 0.01 -bs 16
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=30 -t=ds_test314 -tl=0.7 -lr 0.01 -bs 16
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=30 -t=ds_test315 -tl=0.6 -lr 0.01 -bs 16
