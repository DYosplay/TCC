result="CTL_S11"
for i in {001..010}
do
	PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -lt=compact_triplet_mmd -a 0.96 -b 1.8 -p 0.809 -r 0.297 -lr 0.01 -ep=25 -stop=26 -es=3 -bs=64 -seed=333 -wdb -pf=$result -wpn=$result -t=$result$i -nout=32 -nh=64
done

result="CTL_S12"
for i in {001..010}
do
	PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -lt=compact_triplet_mmd -a 0.96 -b 1.8 -p 0.809 -r 0.297 -lr 0.01 -ep=25 -stop=26 -es=3 -bs=64 -seed=333 -wdb -pf=$result -wpn=$result -t=$result$i -nout=128 -nh=64
done

result="CTL_S13"
for i in {001..010}
do
	PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -lt=compact_triplet_mmd -a 0.96 -b 1.8 -p 0.809 -r 0.297 -lr 0.01 -ep=25 -stop=26 -es=3 -bs=64 -seed=333 -wdb -pf=$result -wpn=$result -t=$result$i -nout=256 -nh=64
done

result="CTL_S14"
for i in {001..010}
do
	PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -lt=compact_triplet_mmd -a 0.96 -b 1.8 -p 0.809 -r 0.297 -lr 0.01 -ep=25 -stop=26 -es=3 -bs=64 -seed=333 -wdb -pf=$result -wpn=$result -t=$result$i -nout=16 -nh=32
done