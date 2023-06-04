PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=ds_test579 -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.5 -q 0.0 -qm 1.0 -dc 0.6 -nlr 0.5 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=ds_test580 -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.5 -q 0.1 -qm 1.5 -dc 0.6 -nlr 0.5 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=ds_test581 -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.5 -q 0.1 -qm 1.5 -dc 0.6 -nlr 0.9 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=ds_test582 -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.5 -q 0.0 -qm 1.5 -dc 0.6 -nlr 0.9 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=ds_test583 -lr 0.01 -bs 32 -lt=triplet_mmd -a 0.0 -tm 1 -p 0.5 -q 0.0 -qm 1.5 -dc 0.6 -nlr 0.5 -stop 26

