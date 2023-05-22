#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=35 -t=ds_test363 -lr 0.01 -bs 16 -lt=quadruplet_loss -qm 0.5
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=35 -t=ds_test364 -lr 0.01 -bs 16 -lt=quadruplet_loss -qm 0.6
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=35 -t=ds_test365 -lr 0.01 -bs 16 -lt=quadruplet_loss -qm 0.7



