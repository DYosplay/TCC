PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=30 -t=ds_test417 -lr 0.001 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.6
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=30 -t=ds_test418 -lr 0.0001 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.7 -q 0.1 -qm 1.5 -dc 0.7
