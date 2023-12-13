PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ctl038 -lr 0.01 -bs 64 -lt=compact_triplet_mmd -b 3 -a 1 -p 0.1 -r 0.1 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -t=ctl038 -ev
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ctl039 -lr 0.01 -bs 32 -lt=compact_triplet_mmd -b 3 -a 1 -p 0.1 -r 0.1 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -t=ctl039 -ev
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0  python run_ds_transformer.py -ep=25 -t=ctl040 -lr 0.01 -bs 64 -lt=compact_triplet_mmd -b 1 -a 1 -p 0.4 -r 0.4 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0  python run_ds_transformer.py -ev -t=ctl040
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0  python run_ds_transformer.py -ep=25 -t=ctl041 -lr 0.01 -bs 64 -lt=compact_triplet_mmd -b 1 -a 1 -p 0.4 -r 0.4 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0  python run_ds_transformer.py -ev -t=ctl041