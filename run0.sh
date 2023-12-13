# for i in {401..420}
# do
#     result="svc_"
#     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=$result$i -lr 0.01 -bs 32 -lt=grl -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.9 -stop 26
# done
# for i in {421..440}
# do
#     result="svc_"
#     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=$result$i -lr 0.01 -bs 32 -lt=grl -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.9 -stop 26 -seed 333
# done

# for i in {041..060}
# do
#     result="svc_"
#     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=$result$i -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.9 -q 0.1 -qm 1.5 -dc 0.9 -stop 26
# done



# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=svc_046 -aw
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=svc_080 -aw
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=svc_422 -aw

# for i in {561..580}
# do
#     result="svc_"
#     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=$result$i -lr 0.01 -bs 32 -lt=triplet_loss -dc 0.9 -stop 26 -z -dsc=finger -seed 333
# done

# for i in {051..060}
# do
#     result="svc_"
#     PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=12 -t=$result$i -aw
# done


# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -t=ctl020 -ev
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ctl030 -lr 0.01 -bs 64 -lt=compact_triplet_mmd -b 3 -a 1 -p 0.1 -r 0.1 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -t=ctl030 -ev
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ctl031 -lr 0.01 -bs 64 -lt=compact_triplet_mmd -b 3 -a 1 -p 0.2 -r 0.2 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ctl032 -lr 0.01 -bs 64 -lt=compact_triplet_mmd -b 3 -a 1 -p 0.3 -r 0.3 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ctl033 -lr 0.01 -bs 64 -lt=compact_triplet_mmd -b 3 -a 1 -p 0.4 -r 0.4 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ctl034 -lr 0.01 -bs 64 -lt=compact_triplet_mmd -b 3 -a 1 -p 0.5 -r 0.5 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ctl035 -lr 0.01 -bs 64 -lt=compact_triplet_mmd -b 3 -a 1 -p 0.6 -r 0.6 -stop 26
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=ctl036 -lr 0.02 -bs 64 -lt=compact_triplet_mmd -b 3 -a 1 -p 0.1 -r 0.1 -stop 26
