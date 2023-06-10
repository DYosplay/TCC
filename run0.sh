# Esses testes serão utilizados na monografia, a normalização de x e y será feita utilizando o zscore e não com relação ao centróide.

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ds_mono01 -lr 0.01 -bs 64 -lt=triplet_loss -dc 0.9 -stop 26 -seed 333 -z
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ds_mono02 -lr 0.01 -bs 64 -lt=triplet_loss -dc 0.9 -stop 26 -seed 888 -z
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ds_mono03 -lr 0.01 -bs 64 -lt=triplet_loss -dc 0.9 -stop 26 -seed 6666 -z
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ds_mono04 -lr 0.01 -bs 64 -lt=triplet_loss -dc 0.9 -stop 26 -seed 9999 -z