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

    if scenario=='finger' : p = np.ones(x.shape)

    x1, y1 = normalize_x_and_y(x, y)

    #result = [x1,y1, zscore(p)]
    result = [x1,y1, p]
    
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
    features = [v, theta, cos, sin, dv, dtheta, logCurRadius, c, totalAccel]

    for f in features:
        result.append(zscore(f))
    ####################################

    return np.array(result)

def pre_process(input_file : str, output_file : str, database : Literal):
    df = None
    if database == MCYT:
        df = pd.read_csv(input_file, sep=' ', header=None, skiprows=1, names=["X", "Y", "TimeStamp", "Uk1", "Uk2", "P"])
    elif database == EBIOSIGN1_DS1 or database == EBIOSIGN1_DS2:
        df = pd.read_csv(input_file, sep=' ', header=None, skiprows=1, names=["X", "Y", "TimeStamp", "P"])
    elif database == BIOSECUR_ID or database == BIOSECURE_DS2:
        df = pd.read_csv(input_file, sep=' ', header=None, skiprows=1, names=["X", "Y", "TimeStamp", "Uk1", "Uk2", "Uk3", "P"])

    x = np.array(df['X'])
    y = np.array(df['Y'])
    p = np.array(df['P'])
    
    dx = diff(x)
    dy = diff(y)
    features = [x,y]
    
    # ele calcula o theta com as derivadas
    # theta = np.arctan2(y, x)
    theta = np.arctan2(dy, dx)
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

    features += [v, theta, cos, sin, p, dv, dtheta, logCurRadius, c, totalAccel]

    result = []

    for f in features:
        result.append(zscore(f))

    buffer = str(len(result[0])) + "\n"

    for i in range(0, len(result[0])):
        buffer += str(result[0][i]) + " " + str(result[1][i]) + " " + str(result[2][i]) + " " + str(result[3][i]) + " " +str(result[4][i]) + " " +str(result[5][i]) + " " +str(result[6][i]) + " " +str(result[7][i]) + " " +str(result[8][i]) + " " +str(result[9][i]) + " " +str(result[10][i]) + " " +str(result[11][i]) + "\n"

    fd = open(output_file, "w")
    fd.write(buffer)
    fd.close()

EBIOSIGN1_DS1 = 0
EBIOSIGN1_DS2 = 1
MCYT = 3
BIOSECUR_ID = 4
BIOSECURE_DS2 = 5
UNDEFINED = -1

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



# if __name__ == '__main__':
#     DEVELOPMENT = ["DeepSignDB/Development/finger", "DeepSignDB/Development/stylus"]
#     EVALUATION  = ["DeepSignDB/Evaluation/finger", "DeepSignDB/Evaluation/stylus"]
    
#     for folder in DEVELOPMENT:
#         print("\n\tWorking on: " + folder)
#         if not os.path.exists("PreProcessed" + os.sep + folder):
#             os.makedirs("PreProcessed" + os.sep + folder)
        
#         files = os.listdir(folder)

#         for file in tqdm(files):
#             input_file  = folder + os.sep + file
#             output_file = "PreProcessed" + os.sep + input_file
#             user_id = int((file.split("_")[0]).split("u")[-1])

#             database = UNDEFINED
#             if user_id >= 1009 and user_id <= 1038:
#                 database = EBIOSIGN1_DS1
#             elif user_id >= 1039 and user_id <= 1084:
#                 database = EBIOSIGN1_DS2
#             elif user_id >= 1 and user_id <= 230:
#                 database = MCYT
#             elif user_id >= 231 and user_id <= 498:
#                 database = BIOSECUR_ID

#             pre_process(input_file, output_file, database)


#     for folder in EVALUATION:
#         print("\n\tWorking on: " + folder)
#         if not os.path.exists("PreProcessed" + os.sep + folder):
#             os.makedirs("PreProcessed" + os.sep + folder)
        
#         files = os.listdir(folder)

#         for file in tqdm(files):
#             input_file  = folder + os.sep + file
#             output_file = "PreProcessed" + os.sep + input_file
#             user_id = int((file.split("_")[0]).split("u")[-1])

#             database = UNDEFINED
#             if user_id >= 373 and user_id <= 407:
#                 database = EBIOSIGN1_DS1
#             elif user_id >= 408 and user_id <= 442:
#                 database = EBIOSIGN1_DS2
#             elif user_id >= 1 and user_id <= 100:
#                 database = MCYT
#             elif user_id >= 101 and user_id <= 232:
#                 database = BIOSECUR_ID
#             elif user_id >= 233 and user_id <= 372:
#                 database = BIOSECURE_DS2

#             pre_process(input_file, output_file, database)