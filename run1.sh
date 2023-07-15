PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=15 -t=ds_triplet_mmd_333 -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.7 -q 0.1 -qm 1.5 -dc 0.9 -stop 26 -seed 333 -r 201.0 -ft -w='epoch10.pt'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=15 -t=ds_triplet_mmd_333 -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.6 -q 0.1 -qm 1.5 -dc 0.9 -stop 26 -seed 333 -r 202.0 -ft -w='epoch10.pt'
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=15 -t=ds_triplet_mmd_333 -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.4 -q 0.1 -qm 1.5 -dc 0.9 -stop 26 -seed 333 -r 203.0 -ft -w='epoch10.pt'

# Fine tuning com finger no cenário 1vs1
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=15 -t=ds_triplet_mmd_333 -lr 0.02 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.9 -stop 26 -seed 333 -r 5.0 -ft -w='epoch10.pt'
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=15 -t=ds_triplet_mmd_333 -lr 0.02 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.8 -stop 26 -seed 333 -r 6.0 -ft -w='epoch10.pt'
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=15 -t=ds_triplet_mmd_333 -lr 0.02 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.7 -stop 26 -seed 333 -r 7.0 -ft -w='epoch10.pt'
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=15 -t=ds_triplet_mmd_333 -lr 0.02 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.6 -stop 26 -seed 333 -r 8.0 -ft -w='epoch10.pt'

# # Daí devo ver qual decaimento deu o melhor resultado e repetir o experimento usando o cenário 4vs1
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=15 -t=ds_triplet_mmd_333 -lr 0.001 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.9 -stop 26 -seed 333 -r 100.0 -ft -w='epoch8.pt'

# Avaliar os mini datasets de finger
# Ainda preciso descobrir qual o melhor peso usando dedo para o cenário 4vs1 e para o cenário 1vs1
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -t=ds_triplet_mmd_333_tuned100.0 -val -w='best.pt' -mode=finger -scene=4vs1 -seed 333
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -t=ds_triplet_mmd_333_tuned0.0 -val -w='epoch12.pt' -mode=finger -scene=4vs1