import numpy as np
import pickle
from tqdm import tqdm
from typing import Dict, List, Tuple
from scipy import stats as st
import numpy as np
import os
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
import math
import DTW.dtw_cuda as dtw
import DTW.soft_dtw_cuda as sdtw
import torch
import sys
from constants import *
dtw_c = dtw.DTW(True, normalize=False, bandwidth=1)

def get_eer(y_true = List[int], y_scores = List[float], result_folder : str = None, generate_graph : bool = False, n_epoch : int = None) -> Tuple[float, float]:
    fpr, tpr, threshold = roc_curve(y_true=y_true, y_score=y_scores, pos_label=1)
    fnr = 1 - tpr

    far = fpr
    frr = fnr

    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # as a sanity check the value should be close to
    eer2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    eer = (eer + eer2)/2
    # eer = min(eer, eer2)

    if generate_graph:
        frr_list = np.array(frr)
        far_list = np.array(far)

        plt.plot(threshold, frr_list, 'b', label="FRR")
        plt.plot(threshold, far_list, 'r', label="FAR")
        plt.legend(loc="upper right")

        plt.xlabel("Threshold")
        plt.ylabel("Error Rate")
        plt.plot(eer_threshold, eer, 'ro')
        plt.text(eer_threshold + 0.05, eer+0.05, s="EER = " + "{:.5f}".format(eer))
        #plt.text(eer_threshold + 1.05, eer2+1.05, s="EER = " + "{:.5f}".format(eer2))
        plt.savefig(result_folder + os.sep + "Epoch" + str(n_epoch) + ".png")
        plt.cla()
        plt.clf()

    return eer, eer_threshold

def new_evaluate(comparison_file : str, n_epoch : int, result_folder : str, tensors_path : str):
    """ Avaliação da rede conforme o arquivo de comparação

    Args:
        comparison_file (str): path do arquivo que contém as assinaturas a serem comparadas entre si, bem como a resposta da comparação. 0 é positivo (original), 1 é negativo (falsificação).
        n_epoch (int): número que indica após qual época de treinamento a avaliação está sendo realizada.
        result_folder (str): path onde salvar os resultados.
    """
    lines = []
    with open(comparison_file, "r") as fr:
        lines = fr.readlines()

    if not os.path.exists(result_folder): os.mkdir(result_folder)

    file_name = (comparison_file.split(os.sep)[-1]).split('.')[0]
    print("\n\tAvaliando " + file_name)
    comparison_folder = result_folder + os.sep + file_name
    if not os.path.exists(comparison_folder): os.mkdir(comparison_folder)

    users = {}

    for line in tqdm(lines, "Calculando distâncias..."):
        distance, user_id, true_label = inference(line, tensors_path=tensors_path)
        
        if user_id not in users: 
            users[user_id] = {"distances": [distance], "true_label": [true_label], "predicted_label": []}
        else:
            users[user_id]["distances"].append(distance)
            users[user_id]["true_label"].append(true_label)

    # Nesse ponto, todas as comparações foram feitas
    buffer = "user, eer_local, threshold, mean_eer, var_th, amp_th, th_range\n"
    local_buffer = ""
    global_true_label = []
    global_distances = []

    eers = []

    local_ths = []

    # Calculo do EER local por usuário:
    for user in tqdm(users, desc="Obtendo EER local..."):
        global_true_label += users[user]["true_label"]
        global_distances  += users[user]["distances"]

        # if "Task" not in comparison_file:
        if 0 in users[user]["true_label"] and 1 in users[user]["true_label"]:
            eer, eer_threshold = get_eer(y_true=users[user]["true_label"], y_scores=users[user]["distances"])
            th_range_local = np.max(np.array(users[user]["distances"])[np.array(users[user]["distances"]) < eer_threshold])

            local_ths.append(eer_threshold)
            eers.append(eer)
            local_buffer += user + ", " + "{:.5f}".format(eer) + ", " + "{:.5f}".format(eer_threshold) + ", 0, 0, 0, " + "{:.5f}".format(eer_threshold -th_range_local) + " (" + "{:.5f}".format(th_range_local) + "~" + "{:.5f}".format(eer_threshold) + ")\n"

    print("Obtendo EER global...")
    
    # Calculo do EER global
    eer_global, eer_threshold_global = get_eer(global_true_label, global_distances, result_folder=comparison_folder, generate_graph=True, n_epoch=n_epoch)

    local_eer_mean = np.mean(np.array(eers))
    local_ths = np.array(local_ths)
    local_ths_var  = np.var(local_ths)
    local_ths_amp  = np.max(local_ths) - np.min(local_ths)
    
    th_range_global = np.max(np.array(global_distances)[np.array(global_distances) < eer_threshold_global])

    buffer += "Global, " + "{:.5f}".format(eer_global) + ", " + "{:.5f}".format(eer_threshold_global) + ", " + "{:.5f}".format(local_eer_mean) + ", " + "{:.5f}".format(local_ths_var) + ", " + "{:.5f}".format(local_ths_amp) + ", " + "{:.5f}".format(eer_threshold_global -th_range_global) + " (" + "{:.5f}".format(th_range_global) + "~" + "{:.5f}".format(eer_threshold_global) + ")\n" + local_buffer

    with open(comparison_folder + os.sep + file_name + " epoch=" + str(n_epoch) + ".csv", "w") as fw:
        fw.write(buffer)

    ret_metrics = {"Global EER": eer_global, "Mean Local EER": local_eer_mean, "Global Threshold": eer_threshold_global, "Local Threshold Variance": local_ths_var, "Local Threshold Amplitude": local_ths_amp}
    print (ret_metrics)
    return ret_metrics

def dte(x, y, len_x, len_y):
    v, matrix = dtw_c(x[None, :int(len_x)], y[None, :int(len_y)]) 
    return v /(64*(len_x + len_y)), matrix

def inference(files : str, tensors_path) -> Tuple[float, str, int]:
    """ Calcula score de dissimilaridade entre um grupo de assinaturas

    Args:
        files (str): nome dos arquivos separados por espaço seguido da label (0: original, 1: falsificada)
        distances (Dict[str, Dict[str, float]]): ex: distances[a][b] = distância entre assinatura a e b.
            distances[a] -> dicionário em que cada chave representa a distância entre a e a chave em questão.
            Esse dicionário contém todas as combinações possíveis entre chaves.
            distances[a][b] == distances[b][a].
    Raises:
        ValueError: caso o formato dos arquivos seja diferente do 4vs1 ou 1vs1.

    Returns:
        Tuple[float, str, int]: score de dissimilaridade, usuário de referência, predição (default: math.nan)
    """
    tokens = files.split(" ")
    user_key = tokens[0].split("_")[0]
    
    result = math.nan
    refs = []
    sign = ""

    if len(tokens) == 3: result = int(tokens[2]); refs.append(tokens[0]); sign = tokens[1]
    elif len(tokens) == 6: result = int(tokens[5]); refs = tokens[0:4]; sign = tokens[4]
    else: raise ValueError("Arquivos de comparação com formato desconhecido")

    s_avg = 0
    s_min = 0
    
    refs_dists = []
    for i in range(0,len(refs)):
        file_path = os.path.join(tensors_path, refs[i].replace(".txt", ".pt"))
        ref_i = torch.load(file_path)  
        for j in range(i+1, len(refs)):
            file_path = os.path.join(tensors_path, refs[j].replace(".txt", ".pt"))
            ref_j = torch.load(file_path) 
            
            value, matrix = dte(ref_i, ref_j, ref_i.shape[0], ref_j.shape[0])
            value = value.detach().cpu().numpy()[0] # libera os gradientes, traz pra cpu, converte em numpy e pega o valor

            refs_dists.append(value)

    refs_dists = np.array(refs_dists)
    
    dists_query = []
    file_path_query = os.path.join(tensors_path, sign.replace(".txt", ".pt"))
    query = torch.load(file_path_query) 
    for i in range(0, len(refs)):
        file_path_ref = os.path.join(tensors_path, refs[i].replace(".txt", ".pt"))
        ref = torch.load(file_path_ref) 

        value, matrix = dte(ref, query, ref.shape[0], query.shape[0])
        value = value.detach().cpu().numpy()[0] # libera os gradientes, traz pra cpu, converte em numpy e pega o valor

        dists_query.append(value)

    dists_query = np.array(dists_query)

    """Cálculos de dissimilaridade a partir daqui"""
    dk = np.mean(refs_dists)
    dk_sqrt = dk ** (1.0/2.0) 

    s_avg = np.sum(dists_query)/dk_sqrt
    s_min = min(dists_query)/dk_sqrt

    return (s_avg + s_min), user_key, result


def main():
    default_path = ".." + os.sep + "OVS" + os.sep + "ROT_X2_" + os.sep + "ROT_X2_005" + os.sep + "generated_features"
    file_path = ""
    if len(sys.argv) < 2:
        print("Usage: python script.py <file_path>")
        print("Using default_path")
        file_path = default_path
    else:
        file_path = sys.argv[1]
    
    print(f"Received file path: {file_path}")

    # Checar arquivo de constantes para protocolos disponíveis
    print(SKILLED_STYLUS_1VS1)
    new_evaluate(SKILLED_STYLUS_1VS1, n_epoch=0, result_folder=".", tensors_path=file_path)

    print(MCYT_SKILLED_4VS1)
    new_evaluate(MCYT_SKILLED_4VS1, n_epoch=0, result_folder=".", tensors_path=file_path)
    
if __name__ == "__main__":
    main()