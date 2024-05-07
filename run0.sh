PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=1 python run_ds_pipeline.py -sig="../Data/DeepSignDB/Evaluation/stylus" --extract_features=Evaluation -seed=333 -t="/Resultados/ROT_X2_/ROT_X2_005/" -pf=".." -w="epoch23.pt"
            
python Investigation/dba.py
python Investigation/dba_distances.py