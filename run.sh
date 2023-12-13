# for i in {001..020}
# do 
#     result="svc_"
#     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=$result$i -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.9 -stop 26 -seed 333 -mix
# done

# for i in {201..220}
# do 
#     result="svc_"
#     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=$result$i -lr 0.01 -bs 32 -lt=triplet_loss -dc 0.9 -stop 26 -seed 333 -z -mix
# done

# for i in {301..320}
# do 
#     result="svc_"
#     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=$result$i -lr 0.01 -bs 32 -lt=triplet_loss -dc 0.9 -stop 26 -seed 333 -z
# done

# for i in {521..540}
# do
#     result="svc_"
#     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=$result$i -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.9 -seed 333 -stop 26 -dsc=finger
# done

# for i in {600..605}
# do
#     result="svc_"
#     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=$result$i -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.9 -stop 26 -seed 333
# done

# Ablacao com triplet_mmd e sem centroide
# for i in {621..630}
# do 
#     result="svc_"
#     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=$result$i -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.9 -stop 26 -seed 333 -z
# done

# # Ablacao sem triplet_mmd e com centroide
# for i in {631..640}
# do 
#     result="svc_"
#     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=$result$i -lr 0.01 -bs 32 -lt=triplet_loss -dc 0.9 -stop 26 -seed 333
# done

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ctl038 -lr 0.01 -bs 64 -lt=compact_triplet_mmd -b 3 -a 1 -p 0.1 -r 0.1 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -t=ctl038 -ev
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ctl039 -lr 0.01 -bs 32 -lt=compact_triplet_mmd -b 3 -a 1 -p 0.1 -r 0.1 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -t=ctl039 -ev
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0  python run_ds_transformer.py -ep=25 -t=ctl040 -lr 0.01 -bs 64 -lt=compact_triplet_mmd -b 1 -a 1 -p 0.4 -r 0.4 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0  python run_ds_transformer.py -ev -t=ctl040
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0  python run_ds_transformer.py -ep=25 -t=ctl041 -lr 0.01 -bs 64 -lt=compact_triplet_mmd -b 1 -a 1 -p 0.4 -r 0.4 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0  python run_ds_transformer.py -ev -t=ctl041
