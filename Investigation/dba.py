# from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import os
import torch
import numpy as np
import re
from tqdm import tqdm
import os

'''
/*******************************************************************************
 * Copyright (C) 2018 Francois Petitjean
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
'''



__author__ ="Francois Petitjean"

def performDBA(series, n_iterations=10):
    n_series = len(series)
    max_length = 0
    for s in series:
        max_length = max(max_length,s.shape[1])

    cost_mat = np.zeros((max_length, max_length))
    delta_mat = np.zeros((max_length, max_length))
    tmp_delta_mat = np.zeros((max_length, max_length))
    path_mat = np.zeros((max_length, max_length), dtype=np.int8)

    medoid_ind = approximate_medoid_index(series,cost_mat,delta_mat,tmp_delta_mat)
    center = series[medoid_ind]

    for i in range(0,n_iterations):
        center = DBA_update(center, series, cost_mat, path_mat, delta_mat,tmp_delta_mat)

    return center

def approximate_medoid_index(series,cost_mat,delta_mat,tmp_delta_mat):
    if len(series)<=50:
        indices = range(0,len(series))
    else:
        indices = np.random.choice(range(0,len(series)),50,replace=False)

    medoid_ind = -1
    best_ss = 1e20
    for index_candidate in indices:
        candidate = series[index_candidate]
        ss = sum_of_squares(candidate,series,cost_mat,delta_mat,tmp_delta_mat)
        if(medoid_ind==-1 or ss<best_ss):
            best_ss = ss
            medoid_ind = index_candidate
    return medoid_ind

def sum_of_squares(s,series,cost_mat,delta_mat,tmp_delta_mat):
    return sum(map(lambda t:squared_DTW(s,t,cost_mat,delta_mat,tmp_delta_mat),series))

def DTW(s,t,cost_mat,delta_mat):
    return np.sqrt(squared_DTW(s,t,cost_mat,delta_mat))

def squared_DTW(s,t,cost_mat,delta_mat,tmp_delta_mat):
    s_len = s.shape[1]
    t_len = t.shape[1]
    fill_delta_mat_dtw(s, t, delta_mat,tmp_delta_mat)
    cost_mat[0, 0] = delta_mat[0, 0]
    for i in range(1, s_len):
        cost_mat[i, 0] = cost_mat[i-1, 0]+delta_mat[i, 0]

    for j in range(1, t_len):
        cost_mat[0, j] = cost_mat[0, j-1]+delta_mat[0, j]

    for i in range(1, s_len):
        for j in range(1, t_len):
            diag,left,top =cost_mat[i-1, j-1], cost_mat[i, j-1], cost_mat[i-1, j]
            if(diag <=left):
                if(diag<=top):
                    res = diag
                else:
                    res = top
            else:
                if(left<=top):
                    res = left
                else:
                    res = top
            cost_mat[i, j] = res+delta_mat[i, j]
    return cost_mat[s_len-1,t_len-1]

def fill_delta_mat_dtw(center, s, delta_mat, tmp_delta_mat):
    n_dims = center.shape[0]
    len_center = center.shape[1]
    len_s=  s.shape[1]
    slim = delta_mat[:len_center,:len_s]
    slim_tmp = tmp_delta_mat[:len_center,:len_s]

    #first dimension - not in the loop to avoid initialisation of delta_mat
    np.subtract.outer(center[0], s[0],out = slim)
    np.square(slim, out=slim)

    for d in range(1,center.shape[0]):
        np.subtract.outer(center[d], s[d],out = slim_tmp)
        np.square(slim_tmp, out=slim_tmp)
        np.add(slim,slim_tmp,out=slim)

    assert(np.abs(np.sum(np.square(center[:,0]-s[:,0]))-delta_mat[0,0])<=1e-6)

def DBA_update(center, series, cost_mat, path_mat, delta_mat, tmp_delta_mat):
    options_argmin = [(-1, -1), (0, -1), (-1, 0)]
    updated_center = np.zeros(center.shape)
    center_length = center.shape[1]
    n_elements = np.zeros(center_length, dtype=int)

    for s in series:
        s_len = s.shape[1]
        fill_delta_mat_dtw(center, s, delta_mat, tmp_delta_mat)
        cost_mat[0, 0] = delta_mat[0, 0]
        path_mat[0, 0] = -1

        for i in range(1, center_length):
            cost_mat[i, 0] = cost_mat[i-1, 0]+delta_mat[i, 0]
            path_mat[i, 0] = 2

        for j in range(1, s_len):
            cost_mat[0, j] = cost_mat[0, j-1]+delta_mat[0, j]
            path_mat[0, j] = 1

        for i in range(1, center_length):
            for j in range(1, s_len):
                diag,left,top =cost_mat[i-1, j-1], cost_mat[i, j-1], cost_mat[i-1, j]
                if(diag <=left):
                    if(diag<=top):
                        res = diag
                        path_mat[i,j] = 0
                    else:
                        res = top
                        path_mat[i,j] = 2
                else:
                    if(left<=top):
                        res = left
                        path_mat[i,j] = 1
                    else:
                        res = top
                        path_mat[i,j] = 2

                cost_mat[i, j] = res+delta_mat[i, j]

        i = center_length-1
        j = s_len-1

        while(path_mat[i, j] != -1):
            updated_center[:,i] += s[:,j]
            n_elements[i] += 1
            move = options_argmin[path_mat[i, j]]
            i += move[0]
            j += move[1]
        assert(i == 0 and j == 0)
        updated_center[:,i] += s[:,j]
        n_elements[i] += 1

    return np.divide(updated_center, n_elements)


# prepara arquivos
# Path to the folder containing the .pt files
if not os.path.exists("DBAs"):
    os.mkdir("DBAs")

folder_path = "Extracted Features" + os.sep + "Evaluation"

# List all files in the folder
file_list = sorted(os.listdir(folder_path))

pattern = re.compile(r'u([0-9][0-9][0-9][0-9])_g.*\.txt')
ignore_sig_type = '_s_'

files = []
mini_files = []

for file_name in file_list:
    
    if file_name.endswith(".pt") and ('v00' in file_name or 'v01' in file_name or 'v02' in file_name or 'v03' in file_name) and ignore_sig_type not in file_name:
        num = int((file_name.split("_")[0]).split('u')[-1])
        if num > 100: continue

        mini_files.append(file_name)
        if len(mini_files) == 4: 
            files.append(mini_files.copy())
            mini_files = []

# Load the PyTorch tensor from the file
for i in tqdm(range(1,1+len(files))):
    tensores = []
    for j in range(len(files[i])):
        tensor = torch.load(os.path.join(folder_path, file_name))
        inverted_tensor = tensor.transpose(0, 1)
        numpy_array = inverted_tensor.detach().cpu().numpy()
        tensores.append(numpy_array)
    tensores = np.array(tensores)

    average_series = performDBA(tensores)
    average_series = (torch.tensor(average_series)).transpose(0, 1)
    torch.save(tensor, "DBAs" + os.sep + "u" + '{:04d}'.format(i) + "_dba_v00.pt")