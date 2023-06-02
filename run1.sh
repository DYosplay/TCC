PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=ds_test563 -lr 0.005 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1.5 -p 0.5 -q 0.1 -qm 1.5 -dc 0.5 -m
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -t=ds_test563 -ev

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=ds_test563 -lr 0.005 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1.5 -p 0.7 -q 0.1 -qm 1.5 -dc 0.5 -m
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -t=ds_test563 -ev

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=ds_test563 -lr 0.005 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.0 -dc 0.5 -m
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -t=ds_test563 -ev

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=ds_test563 -lr 0.005 -bs 32 -lt=triplet_mmd -a 2.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.5 -m
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -t=ds_test563 -ev

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=ds_test563 -lr 0.005 -bs 32 -lt=triplet_mmd -a 0.5 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.5 -m
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -t=ds_test563 -ev

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=ds_test563 -lr 0.005 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1.5 -p 0.8 -q 0.2 -qm 1.5 -dc 0.5 -m
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -t=ds_test563 -ev