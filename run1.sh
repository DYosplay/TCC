PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=ds_test539 -lr 0.005 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.9 -q 0.2 -qm 1.5 -dc 0.6 -nlr 0.01 -ft
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -t=ds_test539 -ev
