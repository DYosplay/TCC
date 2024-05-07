import os
import torch
import numpy as np
from tqdm import tqdm
import pickle
import DTW.dtw_cuda as dtw

dtw_c = dtw.DTW(True, normalize=False, bandwidth=1)

# Function to load a tensor from a file and put it on CUDA
def load_tensor(file_path):
    tensor = torch.load(file_path)
    return tensor.cuda()

# Function to calculate the dte value
def dte(x, y, shape1, shape2):
    # Your dte calculation logic here
    return dtw_c(x[None,], y[None,])[0] / (shape1 + shape2)

# Path to the folders
dbas_folder = "DBAs"
features_folder = "Investigation" + os.sep + "Extracted Features" + os.sep + "Evaluation"

# Initialize the dictionary to store the results
results = {}

# Process files in DBAs folder
for dbas_file_name in tqdm(os.listdir(dbas_folder)):
    if dbas_file_name.endswith(".pt"):
        dbas_tensor = load_tensor(os.path.join(dbas_folder, dbas_file_name))
        dbas_name = dbas_file_name.split("_")[0]
        # dbas_name = 'u' + '{:04d}'.format(int(dbas_name.split("u")[-1]) + 1)
        
        # Process files in generated_features folder
        for features_file_name in tqdm(os.listdir(features_folder)):
            if features_file_name.endswith(".pt") and dbas_name in features_file_name:
                features_tensor = load_tensor(os.path.join(features_folder, features_file_name))
                
                # Calculate shape[1] and shape[2]
                shape1 = dbas_tensor.shape[0]
                shape2 = features_tensor.shape[0]
                
                # Calculate dte value
                dte_value = dte(dbas_tensor, features_tensor, shape1, shape2)
                
                dbas_file_name = dbas_file_name.replace(".pt", ".txt")
                features_file_name = features_file_name.replace(".pt", ".txt")
                # Store in the results dictionary
                if dbas_file_name not in results:
                    results[dbas_file_name] = {}
                results[dbas_file_name][features_file_name] = dte_value
                if features_file_name not in results:
                    results[features_file_name] = {}
                results[features_file_name][dbas_file_name] = dte_value

# Save the dictionary using pickle
with open("dbas.pickle", "wb") as f:
    pickle.dump(results, f)
