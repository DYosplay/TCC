#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=35 -t=ds_test369 -lr 0.001 -bs 16 -lt=quadruplet_loss -qm 4
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=35 -t=ds_test370 -lr 0.001 -bs 16 -lt=quadruplet_loss -qm 6



