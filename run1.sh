#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=35 -t=ds_test366 -lr 0.01 -bs 16 -lt=quadruplet_margin -qm 0.4
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=35 -t=ds_test367 -lr 0.01 -bs 16 -lt=quadruplet_margin -qm 0.3
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=35 -t=ds_test368 -lr 0.01 -bs 16 -lt=quadruplet_margin -qm 0.2