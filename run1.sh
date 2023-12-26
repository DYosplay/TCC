for i in {001..010}
do
    result="CTL_S01_016_"
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_pipeline.py -lt=compact_triplet_mmd -a 0.9 -b 2.7 -p 0.57 -r 0.608 -lr 0.01 -ep=25 -stop=26 -es=3 -bs=64 -seed=333 -wdb -t=$result$i -pf=CTL_S01_016_
done