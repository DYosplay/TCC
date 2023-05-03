import numpy.typing as npt
import numpy as np
import os

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from tqdm import tqdm
import pandas as pd
from scipy.stats import zscore
from scipy import signal

import random

###################################
import dtw_cuda
import torch
import scipy.signal as ssg
from typing import List

import warnings
warnings.filterwarnings("ignore")

################################
EBIOSIGN1_DS1 = 0
EBIOSIGN1_DS2 = 1
MCYT = 3
BIOSECUR_ID = 4
BIOSECURE_DS2 = 5
UNDEFINED = -1

###############################

def resample(sequences:List[npt.ArrayLike], L : int) -> List[npt.ArrayLike]:
    """ Trunca todas as sequencias para que tenham até tamanho L

    Args:
        sequences (List[npt.ArrayLike]): sequências a serem truncadas
        L (int): número de pontos

    Returns:
        npt.ArrayLike: sequências truncadas
    """
    result = []
    for seq in sequences:
        temp = []
        for s in seq:
            temp.append(ssg.resample(s, L))
        # for i in range(0, len(seq)):
        #     temp.append(seq[i][:L])
        
        result.append(temp)

    return result

def traceback2(acc_cost_matrix ):
        """ Encontra a path (matching) dado uma matriz de custo acumulado obtida a partir do cálculo do DTW.
        Args:
            acc_cost_matrix (npt.DTypeLike): matriz de custo acumulado obtida a partir do cálculo do DTW.

        Returns:
            Tuple[npt.ArrayLike, npt.ArrayLike]: coordenadas ponto a ponto referente a primeira e segunda sequência utilizada no cálculo do DTW.
        """
        # acc_cost_matrix = np.transpose(acc_cost_matrix)
        rows, columns = np.shape(acc_cost_matrix)
        rows-=1
        columns-=1

        r = [rows]
        c = [columns]

        while rows != 0 or columns != 0:
            aux = {}
            if rows-1 >= 0: aux[acc_cost_matrix[rows -1][columns]] = (rows - 1, columns)
            if columns-1 >= 0: aux[acc_cost_matrix[rows][columns-1]] = (rows, columns - 1)
            if rows-1 >= 0 and columns-1 >= 0: aux[acc_cost_matrix[rows -1][columns-1]] = (rows -1, columns - 1)
            key = min(aux.keys())
        
            rows, columns = aux[key]

            r.insert(0, rows)
            c.insert(0, columns)
        
        return np.array(r), np.array(c)

def generate_basic_features(input_file : str, scenario : str, database : Literal):
    df = None
    if database == MCYT:
        df = pd.read_csv(input_file, sep=' ', header=None, skiprows=1, names=["X", "Y", "TimeStamp", "Uk1", "Uk2", "P"])
    elif database == EBIOSIGN1_DS1 or database == EBIOSIGN1_DS2:
        df = pd.read_csv(input_file, sep=' ', header=None, skiprows=1, names=["X", "Y", "TimeStamp", "P"])
    elif database == BIOSECUR_ID or database == BIOSECURE_DS2:
        df = pd.read_csv(input_file, sep=' ', header=None, skiprows=1, names=["X", "Y", "TimeStamp", "Uk1", "Uk2", "Uk3", "P"])

    return np.array(df['X']), np.array(df['Y']), np.array(df['P'])

def get_eb_dba(files : List[str], scenario : str, development : bool = True, pre_process : bool = False):
    user_id = int(((files[0].split(os.sep)[-1]).split("_")[0]).split("u")[-1])
    database = get_database(user_id = user_id, scenario=scenario, development=development)
    
    feat = []
    for file in files:
        x,y,p = generate_basic_features(file, scenario, database)
        x1, y1 = normalize_x_and_y(x, y)
        feat.append([x1, y1, zscore(p)])

    features = eb_dba(feat)

    x1 = features[0][:8000]
    y1 = features[1][:8000]
    p = features[2][:8000]

    result = [x1,y1]
    
    ####################################
    dx = diff(x1)
    dy = diff(y1)
    # theta = np.arctan2(dy, dx)
    v = np.sqrt(dx**2+dy**2)
    theta = np.arctan2(dy, dx)
    cos = np.cos(theta)
    sin = np.sin(theta)
    dv = diff(v)
    dtheta = np.abs(diffTheta(theta))
    logCurRadius = np.log((v+0.05) / (dtheta+0.05))
    dv2 = np.abs(v*dtheta)
    totalAccel = np.sqrt(dv**2 + dv2**2)
    c = v * dtheta
    
    features = [v, theta, cos, sin]
    for f in features:
        result.append(zscore(f))

    result.append(p)
    
    features = [dv, dtheta, logCurRadius, c, totalAccel]
    for f in features:
        result.append(zscore(f))

    return np.array(result)


def eb_dba(sequences: List[npt.ArrayLike], distance_measure : str = 'cityblock', timestamp_coord : int = 2) -> npt.ArrayLike:
    """ Calcula a sequência média entre as sequências de entrada usando EB-DBA.

    Args:
        sequences (List[npt.ArrayLike]): K listas de sequências temporais. Dentro de cada lista se encontra a sequência referente a uma feature da sequência temporal.
        Exemplo:
            seq1 = [ [x1], [y1] ]
            seq2 = [ [x2], [y2] ]
            sequences = [ seq1, seq2 ] 
            Nesse caso, o objetivo é calcular o EB-DBA entre as sequências seq1 e seq2 usando DTW dependente com as variáveis x e y das sequências.
            Caso seq1 e seq2 tivessem apenas um elemento, seria o mesmo que calcular o DTW independente.

        distance_measure (str, optional): medida de distância no cálculo do DTW. Defaults to 'cityblock'.
        timestamp_coord (int, optional): indice do conjunto de variáveis onde se encontra o timestamp.

    Returns:
        npt.ArrayLike: EB-DBA entre as sequências
    """

    dtw = dtw_cuda.DTW(True, normalize=False, bandwidth=1)

    # 1) Calcula tamanho médio das sequências
    L = 0
    K = len(sequences)
    L_min = len(sequences[0][0])
    for seq in sequences:
        # validação
        for var in seq:
            if len(seq[0]) != len(var):
                raise ValueError("As sequências possuem tamanhos diferentes entre as variáveis")
        
        L += len(seq[0])
        L_min = min(L_min, len(seq[0]))
    
    L //= len(sequences)
   
    # 2) Resample das frequências usando interpolação linear
    # como não sei como fazer isso, vou truncar as sequências para a menor que tiver
    # L = L_min
    seqs = resample(sequences, L)
    #print("seqs:\n", seqs)
    # 3) Calcula A_EB
    # seqs : List[npt.ArrayLike]
    a_eb = sum(np.array(seqs))/len(seqs)  # media element wise de seqs
    #print("a_eb:\n", a_eb)
    # 4) Calcula A_EB-DBA
    max_t = 1
    calculated_eb_dba = a_eb

    for t in range(0, max_t):
       
        #l_list = []
        #n_list = []
        # Para cada assinatura
        for seq in seqs:
            x = torch.from_numpy(calculated_eb_dba).cuda().transpose(0,1)
            y = torch.from_numpy(np.array(seq)).cuda().transpose(0,1)
            _, path = dtw(x[None,], y[None,])
            path = path[0][1:path.shape[1]-1, 1:path.shape[1]-1].detach().cpu().numpy()
            
            # path são duas listas
            l_list, n_list = traceback2(path)
            if len(l_list) != len(n_list):
                raise ValueError("As listas L e N têm tamanhos diferentes!")

            new_seq = []
            
            for feature_index in range(0, len(seq)):
                temp = {}

                for i in range(0, len(l_list)):
                    l = l_list[i]
                    n = n_list[i]

                    if l not in temp:
                        temp[l] = [seq[feature_index][n]]
                    else:
                        temp[l].append(seq[feature_index][n])   # esse append deveria ser União

                new_seq.append(temp)
            
            new_a_eb_dba = []
            for feature_index in range(0, len(seq)):
                aux = []
                for i in range(0, L):
                    
                    if i not in new_seq[feature_index]:
                        aux.append(0)
                    else:
                        val = np.mean(np.array(new_seq[feature_index][i]))
                        aux.append(val)

                aux = np.array(aux).reshape(-1, 1)
                new_a_eb_dba.append(aux)

        calculated_eb_dba = new_a_eb_dba
             
    return np.array(calculated_eb_dba).reshape(3,L)

#################################
class butterLPFilter(object):
    """docstring for butterLPFilter"""
    def __init__(self, highcut=10.0, fs=100.0, order=3):
        super(butterLPFilter, self).__init__()
        nyq = 0.5 * fs
        highcut = highcut / nyq
        b, a = signal.butter(order, highcut, btype='low')
        self.b = b
        self.a = a
    def __call__(self, data):
        y = signal.filtfilt(self.b, self.a, data)
        return y

def diff(x):
    dx = np.convolve(x, [0.5,0,-0.5], mode='same'); dx[0] = dx[1]; dx[-1] = dx[-2]
    # dx = np.convolve(x, [0.2,0.1,0,-0.1,-0.2], mode='same'); dx[0] = dx[1] = dx[2]; dx[-1] = dx[-2] = dx[-3]
    return dx

def diffTheta(x):
    dx = np.zeros_like(x)
    dx[1:-1] = x[2:] - x[0:-2]; dx[-1] = dx[-2]; dx[0] = dx[1]
    temp = np.where(np.abs(dx)>np.pi)
    dx[temp] -= np.sign(dx[temp]) * 2 * np.pi
    dx *= 0.5
    return dx

def centroid(arr : npt.ArrayLike):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def normalize_x_and_y(x : npt.ArrayLike, y : npt.ArrayLike):
    """ Normaliza as coordenadas x e y de acordo com o centróide

    Args:
        x (npt.ArrayLike): vetor com as coordenadas x
        y (npt.ArrayLike): vetor com as coordenadas y

    Raises:
        ValueError: caso os tamanhos de x e y sejam diferentes

    Returns:
        Tuple[npt.ArrayLike, npt.ArrayLike]: novas coordenadas x, novas coordenadas y
    """
    if(len(x) != len(y)):
        raise ValueError("Tamanhos de x e y diferentes!")

    coord = np.array( list( zip(x, y) ))
    xg, yg = centroid(coord)

    x_hat = np.zeros(len(x))
    x_den = np.max(x) - np.min(x)

    y_hat = np.zeros(len(y))
    y_den = np.max(y) - np.min(y)

    for i in range(0, len(x)):
        x_hat[i] = (x[i] - xg)/x_den

        y_hat[i] = (y[i] - yg)/y_den

    return x_hat, y_hat


bf = butterLPFilter(highcut=15, fs=100)
def generate_features(input_file : str, scenario : str, database : Literal):
    df = None
    if database == MCYT:
        df = pd.read_csv(input_file, sep=' ', header=None, skiprows=1, names=["X", "Y", "TimeStamp", "Uk1", "Uk2", "P"])
    elif database == EBIOSIGN1_DS1 or database == EBIOSIGN1_DS2:
        df = pd.read_csv(input_file, sep=' ', header=None, skiprows=1, names=["X", "Y", "TimeStamp", "P"])
    elif database == BIOSECUR_ID or database == BIOSECURE_DS2:
        df = pd.read_csv(input_file, sep=' ', header=None, skiprows=1, names=["X", "Y", "TimeStamp", "Uk1", "Uk2", "Uk3", "P"])

    p = bf(np.array(df['P']))[:8000]
    x = bf(np.array(df['X']))[:8000]
    y = bf(np.array(df['Y']))[:8000]

    if scenario=='finger' : p = np.ones(x.shape) #* 255

    x1, y1 = normalize_x_and_y(x, y)

    #result = [x1,y1, zscore(p)]
    result = [x1,y1]
    
    ####################################
    dx = diff(x)
    dy = diff(y)
    # theta = np.arctan2(dy, dx)
    v = np.sqrt(dx**2+dy**2)
    theta = np.arctan2(dy, dx)
    cos = np.cos(theta)
    sin = np.sin(theta)
    dv = diff(v)
    dtheta = np.abs(diffTheta(theta))
    logCurRadius = np.log((v+0.05) / (dtheta+0.05))
    dv2 = np.abs(v*dtheta)
    totalAccel = np.sqrt(dv**2 + dv2**2)
    c = v * dtheta
    
    if scenario=='finger': 
        features = [v, theta, cos, sin] 
        features2 = [dv, dtheta, logCurRadius, c, totalAccel]

        for f in features:
            result.append(zscore(f))
        result.append(p)
        for f in features2:
            result.append(zscore(f))
    
    else:
        features = [v, theta, cos, sin, p, dv, dtheta, logCurRadius, c, totalAccel]
        for f in features:
            result.append(zscore(f))
    ####################################

    return np.array(result)

def pre_process(input_file : str, output_file : str, scenario : str, database : Literal):
    df = None
    if database == MCYT:
        df = pd.read_csv(input_file, sep=' ', header=None, skiprows=1, names=["X", "Y", "TimeStamp", "Uk1", "Uk2", "P"])
    elif database == EBIOSIGN1_DS1 or database == EBIOSIGN1_DS2:
        df = pd.read_csv(input_file, sep=' ', header=None, skiprows=1, names=["X", "Y", "TimeStamp", "P"])
    elif database == BIOSECUR_ID or database == BIOSECURE_DS2:
        df = pd.read_csv(input_file, sep=' ', header=None, skiprows=1, names=["X", "Y", "TimeStamp", "Uk1", "Uk2", "Uk3", "P"])

    p = bf(np.array(df['P']))[:8000]
    x = bf(np.array(df['X']))[:8000]
    y = bf(np.array(df['Y']))[:8000]

    if scenario=='finger' : p = np.ones(x.shape) #* 255

    # x1, y1 = normalize_x_and_y(x, y)

    #result = [x1,y1, zscore(p)]
    result = [zscore(x),zscore(y)]
    
    ####################################
    dx = diff(x)
    dy = diff(y)
    # theta = np.arctan2(dy, dx)
    v = np.sqrt(dx**2+dy**2)
    theta = np.arctan2(dy, dx)
    cos = np.cos(theta)
    sin = np.sin(theta)
    dv = diff(v)
    dtheta = np.abs(diffTheta(theta))
    logCurRadius = np.log((v+0.05) / (dtheta+0.05))
    dv2 = np.abs(v*dtheta)
    totalAccel = np.sqrt(dv**2 + dv2**2)
    c = v * dtheta
    
    if scenario=='finger': 
        features = [v, theta, cos, sin] 
        features2 = [dv, dtheta, logCurRadius, c, totalAccel]

        for f in features:
            result.append(zscore(f))
        result.append(p)
        for f in features2:
            result.append(zscore(f))
    
    else: 

        features= np.array([v, theta, cos, sin, p, dv, dtheta, logCurRadius, c, totalAccel])
        features = ((features.transpose() - np.mean(features, axis=1)) / np.std(features, axis=1)).transpose()
        for f in features:
            result.append(f)
            # result.append(zscore(f,axis=1))

    buffer = str(len(result[0])) + "\n"

    sep = ', '

    for i in range(0, len(result[0])):
        buffer += str(result[0][i]) + sep + str(result[1][i]) + sep + str(result[2][i]) + sep + str(result[3][i]) + sep +str(result[4][i]) + sep +str(result[5][i]) + sep +str(result[6][i]) + sep +str(result[7][i]) + sep +str(result[8][i]) + sep +str(result[9][i]) + sep +str(result[10][i]) + sep +str(result[11][i]) + "\n"

    fd = open(output_file, "w")
    fd.write(buffer)
    fd.close()

def get_database(user_id : int, scenario : str, development : bool) -> Literal:
    database = UNDEFINED

    if scenario == 'stylus':
        if development:
            if user_id >= 1009 and user_id <= 1038:
                database = EBIOSIGN1_DS1
            elif user_id >= 1039 and user_id <= 1084:
                database = EBIOSIGN1_DS2
            elif user_id >= 1 and user_id <= 230:
                database = MCYT
            elif user_id >= 231 and user_id <= 498:
                database = BIOSECUR_ID
        else:
            if user_id >= 373 and user_id <= 407:
                database = EBIOSIGN1_DS1
            elif user_id >= 408 and user_id <= 442:
                database = EBIOSIGN1_DS2
            elif user_id >= 1 and user_id <= 100:
                database = MCYT
            elif user_id >= 101 and user_id <= 232:
                database = BIOSECUR_ID
            elif user_id >= 233 and user_id <= 372:
                database = BIOSECURE_DS2

    else:
        if development:
            if user_id >= 1009 and user_id <= 1038:
                database = EBIOSIGN1_DS1
            elif user_id >= 1039 and user_id <= 1084:
                database = EBIOSIGN1_DS2
        else:
            if user_id >= 373 and user_id <= 407:
                database = EBIOSIGN1_DS1
            elif user_id >= 408 and user_id <= 442:
                database = EBIOSIGN1_DS2

    return database

def get_features(file_name : str, scenario : str, development : bool = True):
    user_id = int(((file_name.split(os.sep)[-1]).split("_")[0]).split("u")[-1])
    database = get_database(user_id = user_id, scenario=scenario, development=development)
    return generate_features(file_name, scenario, database)

### augmentation
def get_files(dataset_folder : str):
    files = os.listdir(dataset_folder)
    users = {}
    for file in files:
        tokens = file.split('_')
        key = tokens[0] + tokens[1]
        if key in users:
            users[key].append(dataset_folder + os.sep + file)
        else:
            users[key] = [dataset_folder + os.sep + file]

    return users

def generate_file(folder : str, user : int, sign_state: str, total : int, n_templates : int):
    """_summary_

    Args:
        folder (str): pasta que contém o dataset que passará pela augmentation
        user (int): número do usuário passando pelo processamento
        sign_state (str): 's' para falsificação e 'g' para original
        total (int): total de assinaturas daquele usuário consdierando a condição (genuina ou falsificação) da assinatura
        n_templates (int): números de assinaturas usadas para cada ebdba
    """
    scenario = "stylus"
    database = get_database(user, scenario=scenario, development=True)

    signs = random.sample(list(range(0,total)), total)
    users = get_files(folder)

    for i in range(0, total+1, n_templates):

        file_name = ''
        if database == MCYT:
            file_name = 'u' + f'{user:04d}' + '_' + sign_state + '_' + f'{100 + i:04d}' + 'v' + f'{total + i//n_templates:02d}' + '.txt'
        if database == BIOSECUR_ID:
            file_name = 'u' + f'{user:04d}' + '_' + sign_state + '_u' + f'{100 + i:04d}' + 's0005_sg' + f'{total + i//n_templates:04d}' + '.txt'
        if database == EBIOSIGN1_DS1:
            file_name = 'u' + f'{user:04d}' + '_' + sign_state + '_u' + f'{100 + i:03d}' + '_s3_aug_' + f'{total + i//n_templates:03d}' + '.txt'
        if database == EBIOSIGN1_DS2:
            file_name = 'u' + f'{user:04d}' + '_' + sign_state + '_u' + f'{100 + i:03d}' + '_s3_aug_' + f'{total + i//n_templates:03d}' + '.txt'
          
        signatures = np.array(users['u'+f'{user:04d}' + sign_state])[signs[i:i+n_templates]]

        features = get_eb_dba(signatures, scenario=scenario, development=True)

        buffer = str(len(features[0])) + '\n'

        if database == MCYT:
            uk = " 0 0 0 "
        elif database == EBIOSIGN1_DS1 or database == EBIOSIGN1_DS2:
            uk = " 0 "
        elif database == BIOSECUR_ID or database == BIOSECURE_DS2:
            uk = " 0 0 0 0 "

        for i in range(len(features[0])):
            buffer += str(features[0][i]) + " " + str(features[1][i]) + uk + str(features[2][i]) +'\n'

        with open(folder + os.sep + file_name, 'w') as fw:
            fw.write(buffer)
        
def augmentation(folder : str = "Data/DeepSignDB_Aug/Development/stylus"):
    scenario = 'stylus'
    users = list(range(1, 499)) + list(range(1009, 1085))
    
    for user in tqdm(users): 
        database = get_database(user, scenario=scenario, development=True)
        if database == MCYT:
            generate_file(folder, user, 'g', 25, 5)
            generate_file(folder, user, 's', 25, 5)
        elif database == BIOSECUR_ID:
            generate_file(folder, user, 'g', 16, 4)
            generate_file(folder, user, 's', 12, 4)
        elif database == EBIOSIGN1_DS1 or database == EBIOSIGN1_DS2:
            generate_file(folder, user, 'g', 8, 4)
            generate_file(folder, user, 's', 6, 3)






if __name__ == '__main__':
    # augmentation()
    FOLDER = ["Data/Data Investigation/datasetnp78"]

    # DEVELOPMENT = ["DeepSignDB/Development/finger", "DeepSignDB/Development/stylus"]
    # EVALUATION  = ["DeepSignDB/Evaluation/finger", "DeepSignDB/Evaluation/stylus"]
    # # DEVELOPMENT = ["DeepSignDB/Development/stylus"]
    
    for folder in FOLDER:
        print("\n\tWorking on: " + folder)
        if not os.path.exists("Data/Data Investigation/PreProcessed" + os.sep + folder):
            os.makedirs("Data/Data Investigation/PreProcessed" + os.sep + folder)
        
        files = os.listdir(folder)

        for file in tqdm(files):
            input_file  = folder + os.sep + file
            output_file = "Data/Data Investigation/PreProcessed" + os.sep + input_file + '.csv'
            user_id = int((file.split("_")[0]).split("u")[-1])

            database = UNDEFINED
            if user_id >= 1009 and user_id <= 1038:
                database = EBIOSIGN1_DS1
            elif user_id >= 1039 and user_id <= 1084:
                database = EBIOSIGN1_DS2
            elif user_id >= 1 and user_id <= 230:
                database = MCYT
            elif user_id >= 231 and user_id <= 498:
                database = BIOSECUR_ID

            pre_process(input_file, output_file, 'stylus', database)


    # for folder in EVALUATION:
    #     print("\n\tWorking on: " + folder)
    #     if not os.path.exists("PreProcessed" + os.sep + folder):
    #         os.makedirs("PreProcessed" + os.sep + folder)
        
    #     files = os.listdir(folder)

    #     for file in tqdm(files):
    #         input_file  = folder + os.sep + file
    #         output_file = "PreProcessed" + os.sep + input_file
    #         user_id = int((file.split("_")[0]).split("u")[-1])

    #         database = UNDEFINED
    #         if user_id >= 373 and user_id <= 407:
    #             database = EBIOSIGN1_DS1
    #         elif user_id >= 408 and user_id <= 442:
    #             database = EBIOSIGN1_DS2
    #         elif user_id >= 1 and user_id <= 100:
    #             database = MCYT
    #         elif user_id >= 101 and user_id <= 232:
    #             database = BIOSECUR_ID
    #         elif user_id >= 233 and user_id <= 372:
    #             database = BIOSECURE_DS2

    #         pre_process(input_file, output_file, database)