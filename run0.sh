PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ds_test408 -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1.0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ds_test409 -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1.0
cat "usa sigmoid na triplet loss e nao relu"
