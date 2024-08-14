import random
import numpy as np
import os
import DataLoader.loader as loader
from tqdm import tqdm
from utils.pre_alignment import align
from typing import Dict, Any
import re

def get_files(dataset_folder : str = "../Data/DeepSignDB/Development/stylus"):
    files = os.listdir(dataset_folder)
    users = {}
    for file in files:
        tokens = file.split('_')
        if 'syn' in file:
            key = tokens[0] + "syn" + tokens[1]
        else:
            key = tokens[0] + tokens[1]
        if key in users:
            users[key].append(dataset_folder + os.sep + file)
        else:
            users[key] = [dataset_folder + os.sep + file]

    return users

def files2array(batch, hyperparameters : Dict[str,Any], z : bool, development : bool):
    data = []; lens = []

    for file in batch:
        feat = None
        if '_syn_' in file:
            with open(file,'r') as fr:
                lines = fr.readlines()
                lines = lines[1:]
                feat = np.genfromtxt(lines, delimiter=',').T
        else:
            file = file.replace('\\', os.sep)
            file = file.replace('/', os.sep)
            file = file.strip()
            
            if hyperparameters['signature_path'] is not None:
                file = os.path.join(hyperparameters['signature_path'], file)
                development = 'Development' in file
            else:
                subset_folder = 'Development' if development else "Evaluation"
                file_path = os.path.join(hyperparameters['dataset_folder'], subset_folder, hyperparameters['dataset_scenario'])
                if file_path not in file: file = os.path.join(file_path, file)
            
            feat = loader.get_features(file, hyperparameters=hyperparameters, z=z, development=development)
        data.append(feat)
        lens.append(len(feat[0]))

    max_size = max(lens)

    if hyperparameters['pre_alignment']:
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
def get_batch_from_epoch(epoch, batch_size : int, z : bool, hyperparameters : Dict[str,Any]):
    assert batch_size % 16 == 0
    step = batch_size // 16

    batch = []
    for i in range(0, step):
        batch += epoch.pop()

    data, lens = files2array(batch, hyperparameters=hyperparameters, z=z, development=True)

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

def generate_epoch(dataset_folder : str, hyperparameters : Dict[str, Any], train_offset = [(1, 498), (1009, 1084)], users=None, development = True):
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
    if hyperparameters['rotation']:
        multiplier = 2

    database = None
    print("Gererating new epoch")

    for user_id in tqdm(train_users):
        
        database = loader.get_database(user_id=user_id, development=development, hyperparameters=hyperparameters)

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
            genuines = []
            syn_genuines = []
            s_forgeries = []
            syn_s_forgeries = []
            n_random = 5
            if hyperparameters['synthetic']:
                genuines = random.sample(files['u' + f"{user_id:04}" + 'g'], 5)
                files['u' + f"{user_id:04}" + 'g'] = list(set(files['u' + f"{user_id:04}" + 'g']) - set(genuines))

                syn_genuines = random.sample(files['u' + f"{user_id:04}" + 'syng'], 2)
                files['u' + f"{user_id:04}" + 'g'] = list(set(files['u' + f"{user_id:04}" + 'syng']) - set(syn_genuines))

                s_forgeries = random.sample(files['u' + f"{user_id:04}" + 's'], 4)
                files['u' + f"{user_id:04}" + 's'] = list(set(files['u' + f"{user_id:04}" + 's']) - set(s_forgeries))

                syn_s_forgeries = random.sample(files['u' + f"{user_id:04}" + 'syns'],2)
                files['u' + f"{user_id:04}" + 's'] = list(set(files['u' + f"{user_id:04}" + 'syng']) - set(syn_s_forgeries))

                n_random = 3
            else:
                genuines = random.sample(files['u' + f"{user_id:04}" + 'g'], hyperparameters['ng'])
                files['u' + f"{user_id:04}" + 'g'] = list(set(files['u' + f"{user_id:04}" + 'g']) - set(genuines))

                s_forgeries = random.sample(files['u' + f"{user_id:04}" + 's'], hyperparameters['nf']//2)
                files['u' + f"{user_id:04}" + 's'] = list(set(files['u' + f"{user_id:04}" + 's']) - set(s_forgeries))

            

            # ids aleatórios podem ser de qualquer mini dataset
            random_forgeries_ids = list(set(random.sample(train_users, 6)) - set([user_id]))[:n_random]
            # ids aleatórios apenas do mesmo dataset
            # random_forgeries_ids = get_random_ids(user_id=user_id, database=database, samples=5)

            random_forgeries = []
            for id in random_forgeries_ids:
                random_forgeries.append(random.sample(files_backup['u' + f"{id:04}" + 'g'], 1)[0])

            a = [genuines[0]]
            p = genuines[1:] + syn_genuines
            n = s_forgeries + syn_s_forgeries + random_forgeries

            mini_batch = a + p + n

            epoch.append(mini_batch)
    
    random.shuffle(epoch)
    return epoch