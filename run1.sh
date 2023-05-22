#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=35 -t=ds_test368 -lr 0.01 -bs 16
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=35 -t=ds_test369 -lr 0.01 -bs 16 -a 0.1 -b 0.1
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=35 -t=ds_test370 -lr 0.01 -bs 16 -a 0.3 -b 0.3
