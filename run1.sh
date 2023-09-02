PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python ad_training.py -ep=18 -lf=ds_triplet_mmd_333 -t=adda_006 -lr 0.00001 -bs 16 -stop 26
