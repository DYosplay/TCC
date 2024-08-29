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
import itertools
from scipy.integrate import trapezoid, simpson, cumtrapz,quad
from scipy.interpolate import interp1d

SKILLED_STYLUS_4VS1 = ".." + os.sep + ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_DeepSignDB_skilled_stylus_4vs1.txt"
SKILLED_STYLUS_1VS1 = ".." + os.sep + ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "1vs1" + os.sep + "skilled" + os.sep + "Comp_DeepSignDB_skilled_stylus_1vs1.txt"
RANDOM_STYLUS_4VS1 = ".." + os.sep + ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "random" + os.sep + "Comp_DeepSignDB_random_stylus_4vs1.txt"
RANDOM_STYLUS_1VS1 = ".." + os.sep + ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "1vs1" + os.sep + "random" + os.sep + "Comp_DeepSignDB_random_stylus_1vs1.txt"


def new_evaluate(comparison_file : str, n_epoch : int, result_folder : str, dists_s : Dict[str, float], dists_r : Dict[str, float]):
    """ Avaliação da rede conforme o arquivo de comparação

    Args:
        comparison_file (str): path do arquivo que contém as assinaturas a serem comparadas entre si, bem como a resposta da comparação. 0 é positivo (original), 1 é negativo (falsificação).
        n_epoch (int): número que indica após qual época de treinamento a avaliação está sendo realizada.
        result_folder (str): path onde salvar os resultados.
    """
    lines = []
    with open(comparison_file, "r") as fr:
        lines = fr.readlines()

    os.makedirs(result_folder, exist_ok=True)

    file_name = (comparison_file.split(os.sep)[-1]).split('.')[0]
    print("\n\tAvaliando " + file_name)
    comparison_folder = result_folder + os.sep + file_name
    if not os.path.exists(comparison_folder): os.mkdir(comparison_folder)

    distances_s = np.array(list(sorted(dists_s.values())))
    distances_s = distances_s[distances_s <= 1.0]
    distances_s = distances_s[distances_s >= 0.7]
    distances_r = np.array(list(sorted(dists_r.values())))
    distances_r = distances_r[distances_r <= 0.65]
    distances_r = distances_r[distances_r >= 0.35]
    
    users = {}
    eers = {}
    eers2 = {}

    fa = 0
    fr = 0

    total = len(lines)
    
    for i in tqdm(range(0, len(distances_r)), "Calculando EER..."):
        for j in range(0, len(distances_s)):
            total_true = 0
            total_false = 0

            for line in lines:
                user_id = line.split('_')[0]
                true_label = int(line.split(' ')[-1]) 

                if true_label == 0:
                    total_true += 1
                else:
                    total_false += 1

                if dists_r[line] < distances_r[i]:  
                    # modelo aleatorio considerou assinatura como sendo original
                    
                    if dists_s[line] < distances_s[j]:
                        # modelo profissional considerou assinatura como sendo original
                        if true_label == 1:
                            # porem a assinatura eh falsa
                            fa += 1
                    else:
                        # modelo profissional rejeitou
                        if true_label == 0:
                            # porem a assinatura eh original
                            fr += 1
                else:
                    # modelo aleatorio considerou assinatura como sendo falsa
                    if true_label == 0:
                        # porem a assinatura eh original
                        fr += 1

            ths = (distances_r[i], distances_s[j])
            if ths in eers.keys(): continue

            eers[ths] = abs((fr/total_true) - (fa/total_false))
            eers2[ths]= ((fr/total_true), (fa/total_false))
                        
    min_key = min(eers, key=eers.get)
    print("Diferenca entre ths: " + str(eers[min_key]))
    print("EER variando entre: " + str(eers2[min_key]))

    return

path_s = "dists_s_skilled.pickle"
path_r = "dists_r_skilled.pickle"
# path_s = "dists_s_random.pickle"
# path_r = "dists_r_random.pickle"
dists_r = None
dists_s = None
comparison_file = SKILLED_STYLUS_4VS1
result_folder = "DoubleTH"

with open(path_s, 'rb') as fr:
    dists_s = pickle.load(fr)

with open(path_r, 'rb') as fr:
    dists_r = pickle.load(fr)

new_evaluate(comparison_file, 0, result_folder, dists_s, dists_r)

print()

path_s = "dists_s_random.pickle"
path_r = "dists_r_random.pickle"

dists_r = None
dists_s = None
comparison_file = RANDOM_STYLUS_4VS1
result_folder = "DoubleTH"

with open(path_s, 'rb') as fr:
    dists_s = pickle.load(fr)

with open(path_r, 'rb') as fr:
    dists_r = pickle.load(fr)

new_evaluate(comparison_file, 0, result_folder, dists_s, dists_r)