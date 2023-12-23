# OnlineSignatureVerification

## Busca por hiperparâmetros

### Busca aleatória por hiperparâmetros:
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_pipeline.py -lt=compact_triplet_mmd -lr 0.01 -stop=10 -es=3 -wdb -nt=20 -bs=64 -ep=25 -seed=333 -rs -pf=CTL_S06 -t=CTL_S06

## Treinamentos individuais

### Treino usando Triplet Loss
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -lt=triplet_loss -lr 0.01 -ep=25 -stop=26 -es=3 -bs=64 -seed=333 -wdb -t=Triplet_Loss_001

### Treino usando Triplet MMD
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -lt=triplet_mmd -a 1 -p 0.9 -r 0.1 -lr 0.01 -ep=25 -stop=26 -es=3 -bs=64 -seed=333 -wdb -t=Triplet_MMD_001

### Treino usando Compact Triplet MMD
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -lt=compact_triplet_mmd -a 0.96 -b 1.8 -p 0.809 -r 0.297 -lr 0.01 -ep=25 -stop=26 -es=3 -bs=64 -seed=333 -wdb -t=CTL_002

## Testes

### Avaliar todos os pesos
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -aw -wdb -ep=25 -seed=333 -t=CTL_001

### Avaliar todos os protocolos
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -val -wdb -w best.pt -seed=333 -t=CTL_001

### Avaliar apenas os protocolos do DeepSignDB
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -val -wdb -w best.pt -seed=333 -t=CTL_001

