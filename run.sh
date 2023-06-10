#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=15 -t=ds_mono09 -lr 0.01 -bs 64 -lt=hard_triplet_mmd -dc 0.6 -stop 26 -seed 333 -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=15 -t=ds_mono10 -lr 0.01 -bs 64 -lt=hard_triplet_mmd -dc 0.6 -stop 26 -seed 888 -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=15 -t=ds_mono11 -lr 0.01 -bs 64 -lt=hard_triplet_mmd -dc 0.6 -stop 26 -seed 6666 -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=15 -t=ds_mono12 -lr 0.01 -bs 64 -lt=hard_triplet_mmd -dc 0.6 -stop 26 -seed 9999 -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5