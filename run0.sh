cuda="0"
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=$cuda python run_ds_pipeline.py -ev -w best.pt -seed=333 -t=ROT_X2_005 -pf=ROT_X2_ -wdb
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=$cuda python run_ds_pipeline.py -ev -w epoch23.pt -seed=333 -t=ROT_X2_005 -pf=ROT_X2_ -wdb