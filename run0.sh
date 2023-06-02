PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ds_test556 -lr 0.005 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.6 -m
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -t=ds_test556 -ev

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ds_test557 -lr 0.005 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.9 -m
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -t=ds_test557 -ev

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ds_test558 -lr 0.005 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.8 -q 0.1 -qm 1.5 -dc 0.6 -m
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -t=ds_test558 -ev

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ds_test559 -lr 0.005 -bs 32 -lt=triplet_mmd -a 1.5 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.6 -m
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -t=ds_test559 -ev

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ds_test560 -lr 0.005 -bs 32 -lt=triplet_mmd -a 0.5 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.6 -m
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -t=ds_test560 -ev

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ds_test561 -lr 0.005 -bs 32 -lt=triplet_mmd -a 0.0 -tm 1 -p 0.9 -q 0.0 -qm 1.0 -dc 0.6 -m
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -t=ds_test562 -ev