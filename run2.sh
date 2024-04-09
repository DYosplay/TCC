PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -w epoch23.pt -seed=333 -pf=ROT_X2_ -t=ROT_X2_005 -kgen -p=0 -q=5001 -wdb -wpn="matrix"
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -w epoch23.pt -seed=333 -pf=ROT_X2_ -t=ROT_X2_005 -kgen -p=5001 -q=8697 -wdb -wpn="matrix"
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -w epoch23.pt -seed=333 -pf=ROT_X2_ -t=ROT_X2_005 -kgen -p=8697 -q=15698 -wdb -wpn="matrix"
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -w epoch23.pt -seed=333 -pf=ROT_X2_ -t=ROT_X2_005 -kgen -p=15698 -q=18637 -wdb -wpn="matrix"
