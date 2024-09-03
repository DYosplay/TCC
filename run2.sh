cuda="0"
for i in {001..005}
do
    result="ROT_R11_"
	PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=$cuda python run_ds_pipeline.py -lt=compact_triplet_mmd -a 0.96 -b 1.8 -p 0.809 -r 0.297 -lr 0.01 -ep=25 -stop=26 -es=3 -bs=64 -seed=333 -wdb -pf=$result -wpn=$result -t=$result$i --rotation -ng=5 -nf=5 -r=5
	result="ROT_R12_"
	PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=$cuda python run_ds_pipeline.py -lt=compact_triplet_mmd -a 0.96 -b 1.8 -p 0.809 -r 0.297 -lr 0.01 -ep=25 -stop=26 -es=3 -bs=96 -seed=333 -wdb -pf=$result -wpn=$result -t=$result$i --rotation -ng=5 -nf=5 -nr=13
    result="ROT_R13_"
	PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=$cuda python run_ds_pipeline.py -lt=compact_triplet_mmd -a 0.96 -b 1.8 -p 0.809 -r 0.297 -lr 0.01 -ep=25 -stop=26 -es=3 -bs=128 -seed=333 -wdb -pf=$result -wpn=$result -t=$result$i --rotation -ng=5 -nf=5 -nr=21
done