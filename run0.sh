result="CTL_S06_016_"
for i in {001..035}
do
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=0 python run_ds_pipeline.py -lt=compact_triplet_mmd -a 0.9 -b 2.7 -p 0.57 -r 0.608 -lr 0.01 -ep=25 -stop=26 -es=3 -bs=64 -seed=333 -wdb -t=$result$i -pf=CTL_S06_016_ -wpn=CTL_S06
done

# Gera o sumario dos treinamentos
python GerarMetricas/sumarizar.py ../"$result" Comp_DeepSignDB_skilled_stylus_4vs1
# Gera os boxplots
python GerarMetricas/boxplot.py ../"$result"/resumo_Comp_DeepSignDB_skilled_stylus_4vs1.csv ../"$result"
# Concatena os resultados a fim de criar o csv para o blox plot em mesma escala
python GerarMetricas/concat_files.py "Todos/Resumo Todos.csv" ../"$result"/resumo_Comp_DeepSignDB_skilled_stylus_4vs1.csv
# Gera o boxplot em mesma escala
python GerarMetricas/box2.py "Todos/Resumo Todos.csv" Todos



# # Gera o sumario dos treinamentos
# python3 GerarMetricas/sumarizar.py ../CTL_S01_016_2_ Comp_DeepSignDB_skilled_stylus_4vs1
# # Gera os boxplots
# python3 GerarMetricas/boxplot.py ../CTL_S01_016_2_/resumo_Comp_DeepSignDB_skilled_stylus_4vs1.csv ../CTL_S01_016_2_
# # Concatena os resultados a fim de criar o csv para o blox plot em mesma escala
# python3 GerarMetricas/concat_files.py "Todos/Resumo Todos.csv" ../CTL_S01_016_2_/resumo_Comp_DeepSignDB_skilled_stylus_4vs1.csv
# # Gera o boxplot em mesma escala
# python3 GerarMetricas/box2.py "Todos/Resumo Todos.csv" Todos


# # Gera o sumario dos treinamentos
# python3 GerarMetricas/sumarizar.py ../CTMMD_OFFSET_ Comp_DeepSignDB_skilled_stylus_4vs1
# # Gera os boxplots
# python3 GerarMetricas/boxplot.py ../CTMMD_OFFSET_/resumo_Comp_DeepSignDB_skilled_stylus_4vs1.csv ../CTMMD_OFFSET_
# # Concatena os resultados a fim de criar o csv para o blox plot em mesma escala
# python3 GerarMetricas/concat_files.py "Todos/Resumo Todos.csv" ../CTMMD_OFFSET_/resumo_Comp_DeepSignDB_skilled_stylus_4vs1.csv
# # Gera o boxplot em mesma escala
# python3 GerarMetricas/box2.py "Todos/Resumo Todos.csv" Todos