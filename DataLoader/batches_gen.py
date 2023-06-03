import random
import numpy as np
import os
import DataLoader.loader as loader
from tqdm import tqdm

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

def files2array(batch, scenario : str, developtment : bool):
    data = []; lens = []

    # if developtment:
    #     batch = batch[1:]

    for file in batch:
        if developtment == False: file = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Evaluation" + os.sep + scenario + os.sep + file
        
        # Se quiser testar usando o conjunto de treino
        # if developtment == True: file = "Data" + os.sep + "DeepSignDB" + os.sep + "Development" + os.sep + "stylus" + os.sep + file
        
        feat = loader.get_features(file, scenario=scenario, development=developtment)
        data.append(feat)
        lens.append(len(feat[0]))

    max_size = max(lens)

    generated_batch = []
    for i in range(0, len(data)):
        #resized = resize(data[i], max_size)
        resized = np.pad(data[i], [(0,0),(0,max_size-len(data[i][0]))]) 
        generated_batch.append(resized)

    return np.array(generated_batch), lens

"""DeepSign"""
def get_batch_from_epoch(epoch, batch_size : int, scenario : str):
    assert batch_size % 16 == 0
    step = batch_size // 16

    batch = []
    for i in range(0, step):
        batch += epoch.pop()

    data, lens = files2array(batch, scenario=scenario, developtment=True)

    return data, lens, epoch

def generate_epoch(dataset_folder : str = "../Data/DeepSignDB/Development/stylus", train_offset = [(1, 498), (1009, 1084)], users=None, development = True, scenario : str = 'stylus'):
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

    database = None
    print("Gererating new epoch")

    for user_id in tqdm(train_users):
        
        database = loader.get_database(user_id=user_id, scenario=scenario, development=development)

        if database == loader.EBIOSIGN1_DS1 or database == loader.EBIOSIGN1_DS2:
            number_of_mini_baches = 1
        elif database == loader.MCYT or database == loader.BIOSECURE_DS2:
            number_of_mini_baches = 4
            # continue
        elif database == loader.BIOSECUR_ID:
            number_of_mini_baches = 2
            # continue
        else:
            raise ValueError("Dataset desconhecido!")

        for i in range(0, number_of_mini_baches):

            genuines = random.sample(files['u' + f"{user_id:04}" + 'g'], 6)
            files['u' + f"{user_id:04}" + 'g'] = list(set(files['u' + f"{user_id:04}" + 'g']) - set(genuines))

            s_forgeries = random.sample(files['u' + f"{user_id:04}" + 's'], 5)
            files['u' + f"{user_id:04}" + 's'] = list(set(files['u' + f"{user_id:04}" + 's']) - set(s_forgeries))

            random_forgeries_ids = list(set(random.sample(train_users, 6)) - set([user_id]))[:5]
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