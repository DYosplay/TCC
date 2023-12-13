# for i in {301..320}
# do
#     result="svc_"
#     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=$result$i -lr 0.01 -bs 32 -lt=triplet_loss -dc 0.9 -stop 26 -z
# done
# for i in {321..340}
# do
#     result="svc_"
#     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=$result$i -lr 0.01 -bs 32 -lt=triplet_loss -dc 0.9 -stop 26 -seed 333 -z
# done
# for i in {501..520}
# do
#     result="svc_"
#     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=$result$i -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.9 -stop 26 -dsc=finger
# done

# for i in {061..080}
# do
#     result="svc_"
#     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=$result$i -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.9 -stop 26 -seed 333
# done

# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=svc_314 -aw
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=svc_333 -aw
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=svc_506 -aw
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=svc_413 -aw

# for i in {831..840}
# do
#     result="svc_"
#     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=$result$i -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.7 -q 0.0 -qm 1.5 -dc 0.9 -stop 26
# done

# for i in {841..850}
# do
#     result="svc_"
#     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=$result$i -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.7 -q 0.1 -qm 1.5 -dc 0.9 -stop 26
# done


# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1  python run_ds_transformer.py -ev -t=ctl014
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1  python run_ds_transformer.py -ep=25 -t=ctl024 -lr 0.01 -bs 64 -lt=compact_triplet_mmd -b 1 -a 1 -p 0.4 -r 0.4 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1  python run_ds_transformer.py -ev -t=ctl024
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1  python run_ds_transformer.py -ep=25 -t=ctl025 -lr 0.01 -bs 64 -lt=compact_triplet_mmd -b 1 -a 1 -p 0.5 -r 0.5 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1  python run_ds_transformer.py -ep=25 -t=ctl026 -lr 0.01 -bs 64 -lt=compact_triplet_mmd -b 1 -a 1 -p 0.6 -r 0.6 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1  python run_ds_transformer.py -ep=25 -t=ctl027 -lr 0.01 -bs 64 -lt=compact_triplet_mmd -b 1 -a 1 -p 0.7 -r 0.7 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1  python run_ds_transformer.py -ep=25 -t=ctl028 -lr 0.01 -bs 64 -lt=compact_triplet_mmd -b 1 -a 1 -p 0.8 -r 0.8 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1  python run_ds_transformer.py -ep=25 -t=ctl029 -lr 0.01 -bs 64 -lt=compact_triplet_mmd -b 1 -a 1 -p 1 -r 1 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1  python run_ds_transformer.py -ep=25 -t=ctl037 -lr 0.02 -bs 64 -lt=compact_triplet_mmd -b 1 -a 1 -p 0.4 -r 0.4 -stop 26 