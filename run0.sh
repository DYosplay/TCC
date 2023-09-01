PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python ad_training.py -ep=18 -lf=ds_triplet_mmd_333 -t=adda_004 -lr 0.01 -bs 16 -stop 26

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python ad_training.py -ep=18 -lf=ds_triplet_mmd_333 -t=adda_005 -lr 0.001 -bs 16 -stop 26