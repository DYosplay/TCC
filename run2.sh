for i in {001..010}
do
	result="ROT_S1_"
	PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -lt=compact_triplet_mmd -a 0.96 -b 1.8 -p 0.809 -r 0.297 -lr 0.01 -ep=25 -stop=26 -es=3 -bs=64 -seed=333 -wdb -pf=$result -wpn=$result -t=$result$i
	
	result="ROT_S2_"
	PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -lt=triplet_mmd -a 1 -p 0.9 -r 0.1 -lr 0.01 -ep=25 -stop=26 -es=3 -bs=32 -nw=2 -seed=333 -wdb -pf=$result -wpn=$result -t=$result$i

	result="ROT_S3_"
	PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -lt=triplet_loss -lr 0.01 -ep=25 -stop=26 -es=3 -bs=64 -seed=333 -wdb -pf=$result -wpn=$result -t=$result$i
done