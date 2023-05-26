PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ds_test412 -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1.2 -p 0.1
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ds_test414 -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1.2 -p 0.05
cat "usa sigmoid na triplet loss e nao relu"
