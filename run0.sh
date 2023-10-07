for i in {301..320}
do
    result="svc_"
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_transformer.py -ep=25 -t=$result$i -lr 0.01 -bs 32 -lt=triplet_loss -dc 0.9 -stop 26 -seed 333 -z
done