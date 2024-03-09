import random
import numpy as np
import os
import DataLoader.loader as loader
from tqdm import tqdm
from utils.pre_alignment import align
from typing import Dict, Any

def get_files(dataset_folder : str = "../Data/DeepSignDB/Development/stylus"):
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

def files2array(batch, z : bool, developtment : bool, scenario : str = "stylus", hyperparameters : Dict[str,Any] = None):
    data = []; lens = []
    # scenario = None
    # if developtment:
    #     batch = batch[1:]

    for file in batch:
        file = file.replace('\\', os.sep)
        file = file.replace('/', os.sep)
        
        # print(file)
        if developtment == False and "Evaluation" in file: file = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + file.strip()
        elif developtment == False: file = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Evaluation" + os.sep + scenario + os.sep + file.strip()
        
        # Se quiser testar usando o conjunto de treino
        # if developtment == True: file = "Data" + os.sep + "DeepSignDB" + os.sep + "Development" + os.sep + "stylus" + os.sep + file
        scenario = "stylus" if "stylus" in file.lower() else "finger"

        feat = loader.get_features(file, scenario=scenario, z=z, development=developtment)
        data.append(feat)
        lens.append(len(feat[0]))

    max_size = max(lens)

    if hyperparameters is not None and hyperparameters['pre_alignment']:
        data, lens = align(data,hyperparameters=hyperparameters)
        max_size = max(lens)

    generated_batch = []
    for i in range(0, len(data)):
        #resized = resize(data[i], max_size)
        resized = np.pad(data[i], [(0,0),(0,max_size-len(data[i][0]))]) 
        generated_batch.append(resized)

    return np.array(generated_batch), lens

"""DeepSignMod"""
def transfer_domain_files2array(stylus_batch, finger_batch):
    data = []; lens = []

    for file in stylus_batch:   
        feat = loader.get_features(file, scenario='stylus', development=True)
        data.append(feat)
        lens.append(len(feat[0]))

    for file in finger_batch:   
        feat = loader.get_features(file, scenario='finger', development=True)
        data.append(feat)
        lens.append(len(feat[0]))

    max_size = max(lens)

    generated_batch = []
    for i in range(0, len(data)):
        resized = np.pad(data[i], [(0,0),(0,max_size-len(data[i][0]))]) 
        generated_batch.append(resized)

    return np.array(generated_batch), lens

def generate_transfer_domain_epoch(dataset_folder : str = "../Data/DeepSignMod/Development/", train_offset = [(1009, 1084)]):
    stylus_epoch = generate_epoch(dataset_folder=dataset_folder + 'stylus', train_offset=train_offset, scenario='stylus')
    finger_epoch = generate_epoch(dataset_folder=dataset_folder + 'finger', train_offset=train_offset, scenario='finger')

    return [stylus_epoch, finger_epoch]

def get_batch_from_transfer_domain_epoch(epoch, batch_size : int):
    stylus_epoch, finger_epoch = epoch
    
    assert batch_size % 16 == 0

    stylus_batch = stylus_epoch.pop()
    finger_batch = finger_epoch.pop()

    data, lens = transfer_domain_files2array(stylus_batch, finger_batch)

    epoch = [stylus_epoch, finger_epoch]

    return data, lens, epoch

"""DeepSign"""
def get_batch_from_epoch(epoch, batch_size : int, z : bool, hyperparameters : Dict[str,Any] = None):
    assert batch_size % 16 == 0
    step = batch_size // 16

    batch = []
    for i in range(0, step):
        batch += epoch.pop()

    data, lens = files2array(batch, z=z, developtment=True, hyperparameters=hyperparameters)

    return data, lens, epoch

def get_random_ids(user_id, database, samples = 5):
    if database == loader.EBIOSIGN1_DS1:
        return list(set(random.sample(list(range(1009,1039)), samples+1)) - set([user_id]))[:5]
    elif database == loader.EBIOSIGN1_DS2:
        return list(set(random.sample(list(range(1039,1085)), samples+1)) - set([user_id]))[:5]
    elif database == loader.MCYT:
        return list(set(random.sample(list(range(1,231)), samples+1)) - set([user_id]))[:5]
    elif database == loader.BIOSECUR_ID:
        return list(set(random.sample(list(range(231,499)), samples+1)) - set([user_id]))[:5]
    
    raise ValueError("Dataset desconhecido")

def generate_mixed_epoch(train_offset = [(1, 498), (1009, 1084)]):
    dataset_stylus_folder = "../Data/DeepSignDB/Development/stylus"
    dataset_finger_folder = "../Data/DeepSignDB/Development/finger"
    train_stylus_offset = [(1, 498), (1009, 1084)]
    train_finger_offset = [(1009, 1084)]

    stylus_epoch = generate_epoch(dataset_folder=dataset_stylus_folder, train_offset=train_stylus_offset, scenario='stylus')
    finger_epoch = generate_epoch(dataset_folder=dataset_finger_folder, train_offset=train_finger_offset, scenario='finger')

    epoch = stylus_epoch + finger_epoch
    random.shuffle(epoch)
    return epoch

def generate_epoch(dataset_folder : str, train_offset = [(1, 498), (1009, 1084)], users=None, development = True, scenario : str = 'stylus', hyperparameters = None):
    files = get_files(dataset_folder=dataset_folder)
    files_backup = files.copy()

    train_users = []
    if users is None:
        for t in train_offset:
            train_users += list(range(t[0], t[1]+1))
    else:
        train_users = users

    epoch = []
    number_of_mini_baches = 0

    multiplier = 1
    if hyperparameters is not None and hyperparameters['rotation']:
        multiplier = 2

    database = None
    print("Gererating new epoch")

    for user_id in tqdm(train_users):
        
        database = loader.get_database(user_id=user_id, scenario=scenario, development=development)

        if database == loader.EBIOSIGN1_DS1 or database == loader.EBIOSIGN1_DS2:
            number_of_mini_baches = 1 * multiplier
        elif database == loader.MCYT or database == loader.BIOSECURE_DS2:
            number_of_mini_baches = 4 * multiplier
            # continue
        elif database == loader.BIOSECUR_ID:
            number_of_mini_baches = 2 * multiplier
            # continue
        else:
            raise ValueError("Dataset desconhecido!")

        for i in range(0, number_of_mini_baches):

            genuines = random.sample(files['u' + f"{user_id:04}" + 'g'], 6)
            files['u' + f"{user_id:04}" + 'g'] = list(set(files['u' + f"{user_id:04}" + 'g']) - set(genuines))

            s_forgeries = random.sample(files['u' + f"{user_id:04}" + 's'], 5)
            files['u' + f"{user_id:04}" + 's'] = list(set(files['u' + f"{user_id:04}" + 's']) - set(s_forgeries))

            # ids aleatórios podem ser de qualquer mini dataset
            random_forgeries_ids = list(set(random.sample(train_users, 6)) - set([user_id]))[:5]
            # ids aleatórios apenas do mesmo dataset
            # random_forgeries_ids = get_random_ids(user_id=user_id, database=database, samples=5)

            random_forgeries = []
            for id in random_forgeries_ids:
                random_forgeries.append(random.sample(files_backup['u' + f"{id:04}" + 'g'], 1)[0])

            a = [genuines[0]]
            p = genuines[1:6]
            n = s_forgeries + random_forgeries

            mini_batch = a + p + n

            epoch.append(mini_batch)
    
    random.shuffle(epoch)
    return epoch