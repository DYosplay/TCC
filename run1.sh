#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=35 -t=ds_test371 -lr 0.001 -bs 16 -lt=quadruplet_loss -qm 2 -b 0.2
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=35 -t=ds_test372 -lr 0.001 -bs 16 -lt=quadruplet_loss -qm 6 -b 0.2