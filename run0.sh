aw="_AW"
for i in {001..020}
do
	result="ROT_X1_"
	PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -aw -wdb -ep=25 -seed=333 -pf=$result -t=$result$i -wpn=$result$aw -es=3
done