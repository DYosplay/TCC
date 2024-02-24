from typing import List, Dict, Any
import numpy.typing as npt
import torch
from utils.utilities import traceback
import DTW.dtw_cuda as dtw_cuda
from torch.autograd import Variable
import DTW.soft_dtw_cuda as soft_dtw
import numpy as np


def align(batch : List[npt.ArrayLike], hyperparameters : Dict[str, Any]):
    dtw = dtw_cuda.DTW(True, normalize=False, bandwidth=1)

    batch_size = len(batch)

    ng = hyperparameters['ng']
    nf = hyperparameters['nf']
    nw = hyperparameters['nw']

    assert batch_size % 16 == 0
    step = ng + nf + 1

    lens = []
    aligned_batch = []
    for i in range(0, nw):
        anchor     = torch.tensor(batch[i * step], requires_grad=False).transpose(0,1).cuda()
        signatures = batch[i * step : i * step + step]

        for j in range(0, step):
            signature = torch.tensor(signatures[j], requires_grad=False).transpose(0,1).cuda()

            _, acc_cost_matrix = dtw(anchor[None, :], signature[None, :])
            acc_cost_matrix = torch.squeeze(acc_cost_matrix)
            r,c = traceback(acc_cost_matrix)

            aligned_batch.append(batch[i * step + j].transpose(1,0)[c].transpose(1,0))
            lens.append(len(c))
           
    return aligned_batch, lens

