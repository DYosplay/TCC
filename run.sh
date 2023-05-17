#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=30 -t=ds_test320 -lr 0.001 -bs 16 -lt=cosface
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=30 -t=ds_test321 -lr 0.001 -bs 16 -lt=sphereface
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=30 -t=ds_test322 -lr 0.001 -bs 16 -lt=arcface
