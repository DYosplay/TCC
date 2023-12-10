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

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=ctl007 -lr 0.01 -bs 64 -lt=compact_triplet_loss -b 3 -p 1 -q 0.8 -r 0.4 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=ctl008 -lr 0.01 -bs 64 -lt=compact_triplet_loss -b 2 -p 1 -q 0.8 -r 0.1 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=ctl009 -lr 0.01 -bs 64 -lt=compact_triplet_loss -b 1 -p 1 -q 0.8 -r 0.1 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=ctl010 -lr 0.01 -bs 64 -lt=compact_triplet_loss -b 3 -p 1 -q 0.2 -r 0.2 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=ctl011 -lr 0.01 -bs 64 -lt=compact_triplet_loss -b 2 -p 1 -q 0.2 -r 0.2 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_transformer.py -ep=25 -t=ctl012 -lr 0.01 -bs 64 -lt=compact_triplet_loss -b 1 -p 1 -q 0.2 -r 0.2 -stop 26