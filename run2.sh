result="CTL_S07_016_"
for i in {071..105}
do
	PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -lt=compact_triplet_mmd -a 0.9 -b 2.7 -p 0.57 -r 0.608 -lr 0.01 -ep=25 -stop=26 -es=3 -bs=64 -seed=333 -wdb -t=$result$i -pf=CTL_S07_016_ -wpn=CTL_S07
done