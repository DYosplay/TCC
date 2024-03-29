
from typing import List, Tuple
from numpy import typing as npt

from utils import masked_batch_normalization as mbn
from utils import center_loss as cl
from utils import angular_losses as al
from utils import mmd_loss as mmd
from utils import coral as coral


import torch.nn.utils as nutils
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.optim as optim

from matplotlib import pyplot as plt
import os
import math
from tqdm import tqdm
import numpy as np

import DataLoader.batches_gen as batches_gen
import DTW.soft_dtw_cuda as soft_dtw
import DTW.dtw_cuda as dtw
import utils.metrics as metrics

from fastdtw import fastdtw

import random

CHEAT = False
CHANGE_TRAIN_MODE=80

import warnings
warnings.filterwarnings("ignore")
class DsTransformer(nn.Module):
    def __init__(self, batch_size : int, in_channels : int, dataset_folder : str, gamma : int, lr : float = 0.01, use_mask : bool = False, loss_type : str = 'triplet_loss', alpha : float = 0.0, beta : float = 0.0, p : float = 1.0, q : float = 1.0, r : float = 1.0, qm = 0.5, margin : float = 1.0, decay : int = 0.9, nlr : float = 0.001, use_fdtw : bool = False, fine_tuning: bool = False, early_stop : int = 10, z : bool = False):
        super(DsTransformer, self).__init__()

        # Variáveis do modelo
        self.nw = batch_size//16
        self.ng = 5
        self.nf = 10
        self.margin = margin
        self.quadruplet_margin = qm
        self.model_lambda = 0.01
        self.lr = lr
        self.n_out = 64
        self.n_hidden = 128
        self.n_in = in_channels
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_layers = 2
        self.use_mask = use_mask
        self.loss_type = loss_type
        self.decay = decay
        self.nlr = nlr
        self.use_fdtw = use_fdtw
        self.fine_tuning = fine_tuning
        self.z = z
        # variáveis para a loss
        self.scores = []
        self.labels = []
        self.th = 3.5
        self.alpha = alpha
        self.beta = beta
        self.early_stop = early_stop

        self.p = p
        self.q = q
        self.r = r
        # self.loss_value = math.inf


        # Variáveis que lidam com as métricas/resultados
        self.user_err_avg = 0 
        self.dataset_folder = dataset_folder
        self.comparison_data = {}
        self.buffer = "ComparisonFile, mean_local_eer, global_eer, th_global\n"
        self.eer = []
        self.best_eer = math.inf
        self.last_eer = math.inf
        self.loss_variation = []

        # Definição da rede
        self.cran  = nn.Sequential(
        nn.Conv1d(in_channels=self.n_in, out_channels=self.n_out, kernel_size=7, stride=1, padding=3, bias=True),
        nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True), 
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channels=self.n_out, out_channels=self.n_hidden, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1)
        )
        # self.bn = mbn.MaskedBatchNorm1d(self.n_out)

        self.rnn = nn.GRU(self.n_hidden, self.n_hidden, self.n_layers, dropout=0.1, batch_first=True, bidirectional=False)

        self.h0 = Variable(torch.zeros(self.n_layers, batch_size, self.n_hidden).cuda(), requires_grad=False)
        self.h1 = Variable(torch.zeros(self.n_layers, 5, self.n_hidden).cuda(), requires_grad=False)
        self.h2 = Variable(torch.zeros(self.n_layers, 2, self.n_hidden).cuda(), requires_grad=False)

        # Fecha a update gate (pra virar uma GARU)
        for i in range(self.n_layers):
            eval("self.rnn.bias_hh_l%d"%i)[self.n_hidden:2*self.n_hidden].data.fill_(-1e10) #Initial update gate bias
            eval("self.rnn.bias_ih_l%d"%i)[self.n_hidden:2*self.n_hidden].data.fill_(-1e10) #Initial update gate bias
    
        self.linear = nn.Linear(self.n_hidden, 64, bias=False)

        nn.init.kaiming_normal_(self.linear.weight, a=1) 
        nn.init.kaiming_normal_(self.cran[0].weight, a=0)
        nn.init.kaiming_normal_(self.cran[3].weight, a=0)
        nn.init.zeros_(self.cran[0].bias)
        nn.init.zeros_(self.cran[3].bias)

        self.sdtw = soft_dtw.SoftDTW(True, gamma=5, normalize=False, bandwidth=0.1)
        self.dtw = dtw.DTW(True, normalize=False, bandwidth=1)

        self.center_loss = None
        # self.center_loss = cl.CenterLoss(num_classes=2, feat_dim=1, use_gpu=True)
        self.mmd_loss = mmd.MMDLoss()
        
    def getOutputMask(self, lens):    
        lens = np.array(lens, dtype=np.int32)
        lens = (lens + 1) // 2
        N = len(lens); D = np.max(lens)
        mask = np.zeros((N, D), dtype=np.float32)
        for i in range(N):
            mask[i, 0:lens[i]] = 1.0
        return mask
    
    def forward(self, x, mask, n_epoch):
        length = torch.sum(mask, dim=1)
        length, indices = torch.sort(length, descending=True)
        x = torch.index_select(x, 0, indices)
        mask = torch.index_select(mask, 0, indices)

        h = self.cran(x)
        # h = self.bn(h, length.int())
        h = h.transpose(1,2)
        h = h * mask.unsqueeze(2)

        h = nutils.rnn.pack_padded_sequence(h, list(length.cpu().numpy()), batch_first=True)
        if len(x) == self.batch_size: h, _ = self.rnn(h, self.h0)
        elif len(x) > 2: h, _ = self.rnn(h, self.h1)
        else: h, _ = self.rnn(h, self.h2)
        
        h, length = nutils.rnn.pad_packed_sequence(h, batch_first=True) 
        length = Variable(length).cuda()

        '''Recover the original order'''
        _, indices = torch.sort(indices, descending=False)
        h = torch.index_select(h, 0, indices)
        length = torch.index_select(length, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        
        h = self.linear(h)
        # h = torch.nn.functional.normalize(h, 2)

        if self.training:
            return F.avg_pool1d(h.permute(0,2,1),2,2,ceil_mode=False).permute(0,2,1), (length//2).float()

        return h * mask.unsqueeze(2), length.float()

    def _traceback(self, acc_cost_matrix : npt.DTypeLike):
        """ Encontra a path (matching) dado uma matriz de custo acumulado obtida a partir do cálculo do DTW.
        Args:
            acc_cost_matrix (npt.DTypeLike): matriz de custo acumulado obtida a partir do cálculo do DTW.

        Returns:
            Tuple[npt.ArrayLike, npt.ArrayLike]: coordenadas ponto a ponto referente a primeira e segunda sequência utilizada no cálculo do DTW.
        """
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
    
    def _icnn_loss(self, data, lens):
        """ Loss de um batch

        Args:
            data (torch tensor): Batch composto por mini self.nw mini-batches. Cada mini-batch é composto por 16 assinaturas e se refere a um único usuário e é disposto da seguinte forma:
            [0]: âncora; [1:6]: amostras positivas; [6:11]: falsificações profissionais; [11:16]: aleatórias 
            lens (torch tensor (int)): tamanho de cada assinatura no batch
            n_epoch (int): número da época que se está calculando.

        Returns:
            torch.tensor (float): valor da loss
        """
        step = (self.ng + self.nf + 1)
        total_loss = 0

        # dists = torch.zeros(self.batch_size-self.nw, dtype=data.dtype, device=data.device)
        # dists = torch.zeros(self.ng*self.nw, dtype=data.dtype, device=data.device)

        for i in range(0, self.nw):
            anchor    = data[i * step]
            positives = data[i * step + 1 : i * step + 1 + self.ng] 
            negatives = data[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf]

            len_a = lens[i * step]
            len_p = lens[i * step + 1 : i * step + 1 + self.ng] 
            len_n = lens[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf]

            dist_g = torch.zeros((len(positives)), dtype=data.dtype, device=data.device)
            dist_n = torch.zeros((len(negatives)), dtype=data.dtype, device=data.device)

            '''Average_Pooling_2,4,6'''
            for j in range(len(positives)):
                dist_g[j] = self.sdtw(anchor[None, :int(len_a)], positives[j:j+1, :int(len_p[j])])[0] / (len_a + len_p[j])
                # dists[i*(step-1) + j] = dist_g[j]
                # dists[i*(5) + j] = dist_g[j]

            for j in range(len(negatives)):
                dist_n[j] = self.sdtw(anchor[None, :int(len_a)], negatives[j:j+1, :int(len_n[j])])[0] / (len_a + len_n[j])
                # dists[i*(step-1) + self.ng + j] = dist_n[j]

            # 1. Obter o máximo e mínimo das distâncias
            max_v = torch.max(torch.max(dist_g), torch.max(dist_n))
            min_v = torch.min(torch.min(dist_g), torch.min(dist_n))

            # 2. Normalizar as distâncias das amostras positivas e negativas
            dist_g = (dist_g - min_v) / (max_v - min_v)
            dist_n = (dist_n - min_v) / (max_v - min_v)

            # 3. Calcular lambda_g e lambda_n
            lambda_g = torch.sum(1 - dist_g)
            lambda_n = torch.sum(dist_n)
        
            # 4. Calcular lambda_total
            lambda_total = lambda_g/self.ng + lambda_n/self.nf

            # 5. Calcular omega
            omega = self.alpha - (torch.var(dist_g) + torch.var(dist_n))
            if omega < 0:
                print(omega)
                raise ValueError("omega negativo")

            # 6. Calcular gamma
            gamma = torch.tensor(self.ng / (self.ng + self.nf))
            icnn_score = torch.pow(lambda_total, 1/self.p) + torch.pow(omega, 1/self.q) + torch.pow(gamma, 1/self.r)
            icnn_score /= (self.ng + self.nf)

            total_loss += icnn_score

        return - torch.log2(total_loss)
    
    def _triplet_coral(self, data, lens):
        """ Loss de um batch

        Args:
            data (torch tensor): Batch composto por mini self.nw mini-batches. Cada mini-batch é composto por 16 assinaturas e se refere a um único usuário e é disposto da seguinte forma:
            [0]: âncora; [1:6]: amostras positivas; [6:11]: falsificações profissionais; [11:16]: aleatórias 
            lens (torch tensor (int)): tamanho de cada assinatura no batch
            n_epoch (int): número da época que se está calculando.

        Returns:
            torch.tensor (float): valor da loss
        """
        step = (self.ng + self.nf + 1)
        total_loss = 0
        dists = torch.zeros(self.ng*self.nw, dtype=data.dtype, device=data.device)

        for i in range(0, self.nw):
            anchor    = data[i * step]
            positives = data[i * step + 1 : i * step + 1 + self.ng] 
            negatives = data[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf]

            len_a = lens[i * step]
            len_p = lens[i * step + 1 : i * step + 1 + self.ng] 
            len_n = lens[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf]

            dist_g = torch.zeros((len(positives)), dtype=data.dtype, device=data.device)
            dist_n = torch.zeros((len(negatives)), dtype=data.dtype, device=data.device)

            '''Average_Pooling_2,4,6'''
            for j in range(len(positives)): 
                dist_g[j] = self.sdtw(anchor[None, :int(len_a)], positives[j:j+1, :int(len_p[j])])[0] / (len_a + len_p[j])
                dists[i*self.ng + j] = dist_g[j]

            for j in range(len(negatives)):
                dist_n[j] = self.sdtw(anchor[None, :int(len_a)], negatives[j:j+1, :int(len_n[j])])[0] / (len_a + len_n[j])

            only_pos = torch.sum(dist_g) * (self.model_lambda /self.ng)

            lk = 0
            non_zeros = 1
            for g in dist_g:
                for n in dist_n:
                    temp = F.relu(g + self.margin - n)
                    if temp > 0:
                        lk += temp
                        non_zeros+=1
            lv = lk / non_zeros

            user_loss = lv + only_pos

            total_loss += user_loss
    
        total_loss /= self.nw
       
        # mmd = self.mmd_loss(d1, d2)
        # mmd = F.relu(self.mmd_loss(data[0:self.ng+1+self.nf//2], data[step:step + self.ng+1 + self.nf//2]))
        # mmd1 = F.relu(self.mmd_loss(data[0:step], data[step: step*2]))
        # mmd2 = F.relu(self.mmd_loss(data[step: step*2], data[0:step]))
        cor = torch.tensor(0.0).cuda()
        for i in range(0, step):
            cor += coral.coral(data[i], data[step + i])

        cor *= self.alpha
        var = torch.var(dists) * self.beta
        triplet_loss = total_loss + cor + var

        return triplet_loss
    
    def _norm_triplet_mmd(self, data, lens):
        """ Loss de um batch

        Args:
            data (torch tensor): Batch composto por mini self.nw mini-batches. Cada mini-batch é composto por 16 assinaturas e se refere a um único usuário e é disposto da seguinte forma:
            [0]: âncora; [1:6]: amostras positivas; [6:11]: falsificações profissionais; [11:16]: aleatórias 
            lens (torch tensor (int)): tamanho de cada assinatura no batch
            n_epoch (int): número da época que se está calculando.

        Returns:
            torch.tensor (float): valor da loss
        """

        # data = self.bn(data.transpose(1,2), lens).transpose(1,2)

        step = (self.ng + self.nf + 1)
        total_loss = 0
        dists_gs = torch.zeros(self.ng*self.nw, dtype=data.dtype, device=data.device)
        # dists_ns = torch.zeros(self.nf*self.nw, dtype=data.dtype, device=data.device)

        for i in range(0, self.nw):
            anchor    = data[i * step]
            positives = data[i * step + 1 : i * step + 1 + self.ng] 
            negatives = data[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf]

            len_a = lens[i * step]
            len_p = lens[i * step + 1 : i * step + 1 + self.ng] 
            len_n = lens[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf]

            dist_g = torch.zeros((len(positives)), dtype=data.dtype, device=data.device)
            dist_n = torch.zeros((len(negatives)), dtype=data.dtype, device=data.device)

            '''Average_Pooling_2,4,6'''
            for j in range(len(positives)): 
                dist_g[j] = self.sdtw(anchor[None, :int(len_a)], positives[j:j+1, :int(len_p[j])])[0] / (len_a + len_p[j])
                dists_gs[i*self.ng + j] = dist_g[j]

            for j in range(len(negatives)):
                dist_n[j] = self.sdtw(anchor[None, :int(len_a)], negatives[j:j+1, :int(len_n[j])])[0] / (len_a + len_n[j])
                # dists_ns[i*self.nf + j] = dist_n[j]

            new_dist_g = dist_g - torch.min(dist_g)
            new_dist_n = dist_n - torch.min(dist_g)
            

            only_pos = torch.sum(new_dist_g) * (self.model_lambda /self.ng)

            lk = 0
            non_zeros = 1
            for g in new_dist_g:
                for n in new_dist_n:
                    temp = F.relu(g + self.margin - n)
                    if temp > 0:
                        lk += temp
                        non_zeros+=1
            lv = lk / non_zeros

            user_loss = lv + only_pos

            total_loss += user_loss
    
        total_loss /= self.nw
       
        mmd = self.mmd_loss(data[0:step - 5], data[step: step*2 - 5]) * self.alpha
        
        var_g = torch.var(dists_gs) * self.p
        triplet_loss = total_loss + mmd + var_g

        # var_nr = torch.var(torch.cat([dists_ns[self.nf//2:self.nf], dists_ns[self.nf+self.nf//2:self.nf*2]])) * self.q
        # var_ns = torch.var(torch.cat([dists_ns[0:self.nf//2], dists_ns[self.nf:self.nf+self.nf//2]])) * self.r
        # triplet_loss = total_loss + mmd + var_g + var_nr + var_ns + cor

        return triplet_loss
    
    def _triplet_mmd(self, data, lens):
        """ Loss de um batch

        Args:
            data (torch tensor): Batch composto por mini self.nw mini-batches. Cada mini-batch é composto por 16 assinaturas e se refere a um único usuário e é disposto da seguinte forma:
            [0]: âncora; [1:6]: amostras positivas; [6:11]: falsificações profissionais; [11:16]: aleatórias 
            lens (torch tensor (int)): tamanho de cada assinatura no batch
            n_epoch (int): número da época que se está calculando.

        Returns:
            torch.tensor (float): valor da loss
        """

        # data = self.bn(data.transpose(1,2), lens).transpose(1,2)

        step = (self.ng + self.nf + 1)
        total_loss = 0
        dists_gs = torch.zeros(self.ng*self.nw, dtype=data.dtype, device=data.device)
        dists_ns = torch.zeros(self.nf*self.nw, dtype=data.dtype, device=data.device)

        for i in range(0, self.nw):
            anchor    = data[i * step]
            positives = data[i * step + 1 : i * step + 1 + self.ng] 
            negatives = data[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf]

            len_a = lens[i * step]
            len_p = lens[i * step + 1 : i * step + 1 + self.ng] 
            len_n = lens[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf]

            dist_g = torch.zeros((len(positives)), dtype=data.dtype, device=data.device)
            dist_n = torch.zeros((len(negatives)), dtype=data.dtype, device=data.device)

            '''Average_Pooling_2,4,6'''
            for j in range(len(positives)): 
                dist_g[j] = self.sdtw(anchor[None, :int(len_a)], positives[j:j+1, :int(len_p[j])])[0] / (len_a + len_p[j])
                dists_gs[i*self.ng + j] = dist_g[j]

            for j in range(len(negatives)):
                dist_n[j] = self.sdtw(anchor[None, :int(len_a)], negatives[j:j+1, :int(len_n[j])])[0] / (len_a + len_n[j])
                dists_ns[i*self.nf + j] = dist_n[j]

            only_pos = torch.sum(dist_g) * (self.model_lambda /self.ng)

            # lk = 0
            # non_zeros = 1
            # for g in dist_g:
            #     for n in dist_n[:5]:
            #         temp = F.relu(g + self.margin - n) * self.p
            #         if temp > 0:
            #             lk += temp
            #             non_zeros+=1

            #     for n in dist_n[5:]:
            #         temp = F.relu(g + self.quadruplet_margin - n) * (1 - self.p)
            #         if temp > 0:
            #             lk += temp
            #             non_zeros+=1
            # lv = lk / non_zeros
            lk1 = F.relu(dist_g.unsqueeze(1) + self.margin - dist_n[:5].unsqueeze(0)) * self.p
            lk2 = F.relu(dist_g.unsqueeze(1) + self.margin - dist_n[5:].unsqueeze(0)) * (1-self.p)
            lv = (torch.sum(lk1) + torch.sum(lk2)) / (lk1.data.nonzero(as_tuple=False).size(0) + lk2.data.nonzero(as_tuple=False).size(0) + 1)

            user_loss = lv + only_pos

            total_loss += user_loss
    
        total_loss /= self.nw
       
        mmd1 = self.mmd_loss(data[0:step - 5], data[step: step*2 - 5]) * self.alpha
        var_g = torch.var(dists_gs) * self.q
        # var_nr = torch.var(torch.cat([dists_ns[self.nf//2:self.nf], dists_ns[self.nf+self.nf//2:self.nf*2]])) * self.q
        # var_ns = torch.var(torch.cat([dists_ns[0:self.nf//2], dists_ns[self.nf:self.nf+self.nf//2]])) * self.r
        # triplet_loss = total_loss + mmd + var_g + var_nr + var_ns + cor

        # ampli = torch.abs(torch.max(dists_gs) - torch.min(dists_gs)) * self.q
 
        return total_loss + mmd1 + var_g

    def _hard_triplet_mmd(self, data, lens):
        """ Loss de um batch

        Args:
            data (torch tensor): Batch composto por mini self.nw mini-batches. Cada mini-batch é composto por 16 assinaturas e se refere a um único usuário e é disposto da seguinte forma:
            [0]: âncora; [1:6]: amostras positivas; [6:11]: falsificações profissionais; [11:16]: aleatórias 
            lens (torch tensor (int)): tamanho de cada assinatura no batch
            n_epoch (int): número da época que se está calculando.

        Returns:
            torch.tensor (float): valor da loss
        """

        # data = self.bn(data.transpose(1,2), lens).transpose(1,2)

        step = (self.ng + self.nf + 1)
        total_loss = 0
        dists_gs = torch.zeros(self.ng*self.nw, dtype=data.dtype, device=data.device)
        dists_ns = torch.zeros(self.nf*self.nw, dtype=data.dtype, device=data.device)

        mmd_loss = torch.tensor(0.0, device=data.device)

        for i in range(0, self.nw):
            anchor    = data[i * step]
            positives = data[i * step + 1 : i * step + 1 + self.ng] 
            negatives = data[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf]

            len_a = lens[i * step]
            len_p = lens[i * step + 1 : i * step + 1 + self.ng] 
            len_n = lens[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf]

            dist_g = torch.zeros((len(positives)), dtype=data.dtype, device=data.device)
            dist_n = torch.zeros((len(negatives)), dtype=data.dtype, device=data.device)

            '''Average_Pooling_2,4,6'''
            for j in range(len(positives)): 
                dist_g[j] = self.sdtw(anchor[None, :int(len_a)], positives[j:j+1, :int(len_p[j])])[0] / (len_a + len_p[j])
                dists_gs[i*self.ng + j] = dist_g[j]

            for j in range(len(negatives)):
                dist_n[j] = self.sdtw(anchor[None, :int(len_a)], negatives[j:j+1, :int(len_n[j])])[0] / (len_a + len_n[j])
                dists_ns[i*self.nf + j] = dist_n[j]

            only_pos = torch.sum(dist_g) * (self.model_lambda /self.ng)

            lk1 = F.relu(dist_g.unsqueeze(1) + self.margin - dist_n[:5].unsqueeze(0)) * self.p
            lk2 = F.relu(dist_g.unsqueeze(1) + self.margin - dist_n[5:].unsqueeze(0)) * (1-self.p)
            lv = (torch.sum(lk1) + torch.sum(lk2)) / (lk1.data.nonzero(as_tuple=False).size(0) + lk2.data.nonzero(as_tuple=False).size(0) + 1)

            user_loss = lv + only_pos

            total_loss += user_loss
    
        total_loss /= self.nw

        comparisons = 0
        for i in range(self.nw-1):
            m = torch.max(lens[i*step:(i+1)*step - 5])
            for j in range(i+1,self.nw):
                n = torch.max(lens[j*step:(j+1)*step - 5])
                trunc = int(torch.max(m,n))

                mmd_loss += self.mmd_loss(data[i*step:(i+1)*step - 5, :trunc], data[j*step:(j+1)*step - 5, :trunc]) * self.alpha
                comparisons +=1
                
        var_g = torch.var(dists_gs) * self.q
        
        
 
        return total_loss + (mmd_loss/comparisons) + var_g         

    def _loss(self, data, lens):
        """ Loss de um batch

        Args:
            data (torch tensor): Batch composto por mini self.nw mini-batches. Cada mini-batch é composto por 16 assinaturas e se refere a um único usuário e é disposto da seguinte forma:
            [0]: âncora; [1:6]: amostras positivas; [6:11]: falsificações profissionais; [11:16]: aleatórias 
            lens (torch tensor (int)): tamanho de cada assinatura no batch
            n_epoch (int): número da época que se está calculando.

        Returns:
            torch.tensor (float): valor da loss
        """
        step = (self.ng + self.nf + 1)
        total_loss = 0

        # dists = torch.zeros(self.batch_size-self.nw, dtype=data.dtype, device=data.device)
        # dists = torch.zeros(self.ng*self.nw, dtype=data.dtype, device=data.device)

        for i in range(0, self.nw):
            anchor    = data[i * step]
            positives = data[i * step + 1 : i * step + 1 + self.ng] 
            negatives = data[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf]

            len_a = lens[i * step]
            len_p = lens[i * step + 1 : i * step + 1 + self.ng] 
            len_n = lens[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf]

            dist_g = torch.zeros((len(positives)), dtype=data.dtype, device=data.device)
            dist_n = torch.zeros((len(negatives)), dtype=data.dtype, device=data.device)

            '''Average_Pooling_2,4,6'''
            for j in range(len(positives)): 
                dist_g[j] = self.sdtw(anchor[None, :int(len_a)], positives[j:j+1, :int(len_p[j])])[0] / (len_a + len_p[j])
                # dists[i*(step-1) + j] = dist_g[j]
                # dists[i*(5) + j] = dist_g[j]

            for j in range(len(negatives)):
                dist_n[j] = self.sdtw(anchor[None, :int(len_a)], negatives[j:j+1, :int(len_n[j])])[0] / (len_a + len_n[j])
                # dists[i*(step-1) + self.ng + j] = dist_n[j]
                
            only_pos = torch.sum(dist_g) * (self.model_lambda /self.ng)
            # var_g = torch.var(dist_g) * self.alpha
            # var_n = torch.var(dist_n) * self.beta

            lk = 0
            non_zeros = 1
            for g in dist_g:
                for n in dist_n:
                    temp = F.relu(g + self.margin - n)
                    if temp > 0:
                        lk += temp
                        non_zeros+=1
            lv = lk / non_zeros

            # lk = torch.sum(F.relu(dist_g.unsqueeze(1) + self.margin - dist_n.unsqueeze(0)))
            # lv = lk / (lk.data.nonzero(as_tuple=False).size(0) + 1)

            user_loss = lv + only_pos #+ var_g + var_n

            total_loss += user_loss
    
        total_loss /= self.nw
        triplet_loss = total_loss

        return triplet_loss
    
    def _dte(self, x, y, len_x, len_y):
        """ DTW entre assinaturas x e y normalizado pelos seus tamanhos * dimensões

        Args:
            x (torch.tensor): Assinatura 1
            y (torch.tensor): Assinatura 2
            len_x (int): Tamanho da assinatura 1
            len_y (int): Tamanho da assinatura 2

        Returns:
            float: DTW normalizado entre as assinaturas
        """
        if self.use_fdtw:
            distance, _ = fastdtw(x[:int(len_x)].detach().cpu().numpy(), y[:int(len_y)].detach().cpu().numpy(), dist=2)
            return torch.tensor([distance /(64* (len_x + len_y))])
        else:
            return self.dtw(x[None, :int(len_x)], y[None, :int(len_y)])[0] /(64* (len_x + len_y))

    def _inference(self, files : str, scenario : str, n_epoch : int) -> Tuple[float, str, int]:
        """
        Args:
            files (str): string no formato: ref1 [,ref2, ref3, ref4], sign, label 

        Raises:
            ValueError: "Arquivos de comparação com formato desconhecido"

        Returns:
            float, str, int: distância da assinatura, usuário, label 
        """
        tokens = files.split(" ")
        user_key = tokens[0].split("_")[0]
        
        result = math.nan
        refs = []
        sign = ""

        if len(tokens) == 3: result = int(tokens[2]); refs.append(tokens[0]); sign = tokens[1]
        elif len(tokens) == 6: result = int(tokens[5]); refs = tokens[0:4]; sign = tokens[4]
        else: raise ValueError("Arquivos de comparação com formato desconhecido")

        test_batch, lens = batches_gen.files2array(refs + [sign], z=self.z, scenario=scenario, developtment=CHEAT)

        mask = self.getOutputMask(lens)
        
        mask = Variable(torch.from_numpy(mask)).cuda()
        inputs = Variable(torch.from_numpy(test_batch)).cuda()

        embeddings, lengths = self(inputs.float(), mask, n_epoch)    
        refs = embeddings[:len(embeddings)-1]
        sign = embeddings[-1]

        len_refs = lengths[:len(embeddings)-1]
        len_sign = lengths[-1]

        dk_list = []
        dk = math.nan
        count = 0
        if len(refs) == 1 : dk = 1
        else:
            dk = 0
            for i in range(0, len(refs)):
                for j in range(1, len(refs)):
                    if i < j:
                        dk += (self._dte(refs[i], refs[j], len_refs[i], len_refs[j]))
                        count += 1

            dk = dk/(count)
    
        dk_sqrt = math.sqrt(dk)
        
        dists = []
        for i in range(0, len(refs)):
            dists.append(self._dte(refs[i], sign, len_refs[i], len_sign).detach().cpu().numpy()[0])

        dists = np.array(dists) / dk_sqrt
        s_avg = np.mean(dists)
        s_min = min(dists)

        return s_avg + s_min, user_key, result

    def new_evaluate(self, comparison_file : str, n_epoch : int, result_folder : str):
        """ Avaliação da rede conforme o arquivo de comparação

        Args:
            comparison_file (str): path do arquivo que contém as assinaturas a serem comparadas entre si, bem como a resposta da comparação. 0 é positivo (original), 1 é negativo (falsificação).
            n_epoch (int): número que indica após qual época de treinamento a avaliação está sendo realizada.
            result_folder (str): path onde salvar os resultados.
        """

        self.train(mode=False)
        lines = []
        with open(comparison_file, "r") as fr:
            lines = fr.readlines()

        scenario = 'stylus'
        if 'finger' in comparison_file:
            scenario = 'finger'

        if not os.path.exists(result_folder): os.mkdir(result_folder)

        file_name = (comparison_file.split(os.sep)[-1]).split('.')[0]
        print("\n\tAvaliando " + file_name)
        comparison_folder = result_folder + os.sep + file_name
        if not os.path.exists(comparison_folder): os.mkdir(comparison_folder)

        users = {}

        for line in tqdm(lines, "Calculando distâncias..."):
            distance, user_id, true_label = self._inference(line, scenario=scenario, n_epoch=n_epoch)
            
            if user_id not in users: 
                users[user_id] = {"distances": [distance], "true_label": [true_label], "predicted_label": []}
            else:
                users[user_id]["distances"].append(distance)
                users[user_id]["true_label"].append(true_label)

        # Nesse ponto, todos as comparações foram feitas
        buffer = "user, eer_local, threshold\n"
        local_buffer = ""
        global_true_label = []
        global_distances = []

        eers = []

        # Calculo do EER local por usuário:
        for user in tqdm(users, desc="Obtendo EER local..."):
            global_true_label += users[user]["true_label"]
            global_distances  += users[user]["distances"]

            eer, eer_threshold = metrics.get_eer(y_true=users[user]["true_label"], y_scores=users[user]["distances"])
            eers.append(eer)
            local_buffer += user + ", " + "{:.5f}".format(eer) + ", " + "{:.5f}".format(eer_threshold) + "\n"

        print("Obtendo EER global...")
        
        # Calculo do EER global
        eer_global, eer_threshold_global = metrics.get_eer(global_true_label, global_distances, result_folder=comparison_folder, generate_graph=True, n_epoch=n_epoch)

        buffer += "Global, " + "{:.5f}".format(eer_global) + ", " + "{:.5f}".format(eer_threshold_global) + "\n" + local_buffer

        local_eer_mean = np.mean(np.array(eers))
        self.buffer += file_name + ", " + "{:.5f}".format(local_eer_mean) + ", " + "{:.5f}".format(eer_global) + ", " + "{:.5f}".format(eer_threshold_global) + "\n"

        with open(comparison_folder + os.sep + file_name + " epoch=" + str(n_epoch) + ".csv", "w") as fw:
            fw.write(buffer)

        self.last_eer = eer_global

        if n_epoch != 0 and n_epoch != 777 and n_epoch != 888:
            if eer_global < self.best_eer:
                torch.save(self.state_dict(), result_folder + os.sep + "Backup" + os.sep + "best.pt")
                self.best_eer = eer_global
                print("EER atualizado: " + str(self.best_eer))

        self.train(mode=True)

    def start_train(self, n_epochs : int, batch_size : int, comparison_files : List[str], result_folder : str, triplet_loss_w : float = 0.5, fine_tuning : bool = False):
        """ Loop de treinamento

        Args:
            n_epochs (int): Número de épocas que o treinamento deve ocorrer
            batch_size (int): Tamanho do batch de treinamento
            comparison_files (List[str]): Lista com as paths dos arquivos de comparação a serem avaliados durante o treinamento.
            result_folder (str): Path de onde os resultados de avaliação e o backup dos pesos devem ser armazenados.
        """
        optimizer = None
        lr_scheduler = None
        flag = False
        if self.fine_tuning: flag = True

        if self.loss_type == 'icnn_loss':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.decay) 
        elif self.loss_type == 'triplet_loss' or self.loss_type == 'quadruplet_loss' or self.loss_type == 'hard_triplet_mmd' or self.loss_type == 'triplet_mmd' or self.loss_type == 'triplet_coral' or self.loss_type == 'norm_triplet_mmd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.decay) 
        
        losses = [math.inf]*10

        running_loss = 0

        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        if not os.path.exists(result_folder + os.sep + "Backup"):
            os.mkdir(result_folder + os.sep + "Backup")
        bckp_path = result_folder + os.sep + "Backup"

        for i in range(1, n_epochs+1):
            epoch = None
            scenario = None
            if not self.fine_tuning:
                epoch = batches_gen.generate_epoch()
                scenario = 'stylus'
            else:
                epoch = batches_gen.generate_epoch(dataset_folder="../Data/DeepSignDB/Development/finger", train_offset=[(1009, 1084)], scenario='finger')
                scenario = 'finger'

            epoch_size = len(epoch)
            self.loss_value = running_loss/epoch_size
            losses.append(self.loss_value)
            losses = losses[1:]

            if not self.fine_tuning and ((self.best_eer > 0.025 and i >= self.early_stop) or (self.loss_value > min(losses) and i > 50)):
                print("\n\nEarly stop!")
                break

            pbar = tqdm(total=(epoch_size//(batch_size//16)), position=0, leave=True, desc="Epoch " + str(i) +" PAL: " + "{:.3f}".format(self.loss_value))

            running_loss = 0
            self.mean_eer = 0
            #PAL = Previous Accumulated Loss
            while epoch != []:
                batch, lens, epoch = batches_gen.get_batch_from_epoch(epoch, batch_size, z=self.z, scenario=scenario)
                
                mask = self.getOutputMask(lens)

                mask = Variable(torch.from_numpy(mask)).cuda()
                inputs = Variable(torch.from_numpy(batch)).cuda()
                
                
                outputs, length = self(inputs.float(), mask, i)
                optimizer.zero_grad()
                loss = None

                if self.loss_type == 'hard_triplet_mmd':
                    loss = self._hard_triplet_mmd(outputs, length)
                    loss.backward()
                elif self.loss_type == 'triplet_mmd':
                    loss = self._triplet_mmd(outputs, length)
                    loss.backward()
                elif self.loss_type == 'norm_triplet_mmd':
                    loss = self._norm_triplet_mmd(outputs, length)
                    loss.backward()
                elif self.loss_type == 'triplet_coral':
                    loss = self._triplet_coral(outputs, length)
                    loss.backward()
                elif self.loss_type == 'quadruplet_loss':
                    loss = self._quadruplet_loss(outputs, length)
                    loss.backward()
                elif self.loss_type == 'icnn_loss':
                    loss = self._icnn_loss(outputs, length)
                    loss.backward()
                else:
                    loss = self._loss(outputs, length)
                    loss.backward()

                optimizer.step()

                running_loss += loss.item()
            
                pbar.update(1)

            pbar.close()

            if fine_tuning or i >= CHANGE_TRAIN_MODE or (i % 5 == 0 or i > (n_epochs - 3) ):
                for cf in comparison_files:
                    self.new_evaluate(comparison_file=cf, n_epoch=i, result_folder=result_folder)
            
            self.loss_variation.append(running_loss/epoch_size)
            
            lr_scheduler.step()

            torch.save(self.state_dict(), bckp_path + os.sep + "epoch" + str(i) + ".pt")

        # Loss graph
        plt.xlabel("#Epoch")
        plt.ylabel("Loss")
        plt.plot(list(range(0,len(self.loss_variation))), self.loss_variation)
        plt.savefig(result_folder + os.sep + "loss.png")
        plt.cla()
        plt.clf()

    def transfer_domain(self, n_epochs : int, batch_size : int, comparison_files : List[str], result_folder : str):
        """ Loop de treinamento

        Args:
            n_epochs (int): Número de épocas que o treinamento deve ocorrer
            batch_size (int): Tamanho do batch de treinamento
            comparison_files (List[str]): Lista com as paths dos arquivos de comparação a serem avaliados durante o treinamento.
            result_folder (str): Path de onde os resultados de avaliação e o backup dos pesos devem ser armazenados.
        """
        optimizer = None
        lr_scheduler = None
        flag = False
        if self.fine_tuning: flag = True

        if self.loss_type == 'icnn_loss':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.decay) 
        elif self.loss_type == 'triplet_loss' or self.loss_type == 'quadruplet_loss' or self.loss_type == 'hard_triplet_mmd' or self.loss_type == 'triplet_mmd' or self.loss_type == 'triplet_coral' or self.loss_type == 'norm_triplet_mmd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.decay) 
        
        losses = [math.inf]*10

        running_loss = 0

        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        if not os.path.exists(result_folder + os.sep + "Backup"):
            os.mkdir(result_folder + os.sep + "Backup")
        bckp_path = result_folder + os.sep + "Backup"

        for i in range(1, n_epochs+1):
            epoch = batches_gen.generate_transfer_domain_epoch()
            epoch_size = min(len(epoch[0]), len(epoch[1]))
            self.loss_value = running_loss/epoch_size
            losses.append(self.loss_value)
            losses = losses[1:]

            pbar = tqdm(total=epoch_size, position=0, leave=True, desc="Epoch " + str(i) +" PAL: " + "{:.3f}".format(self.loss_value))

            running_loss = 0
            self.mean_eer = 0
            #PAL = Previous Accumulated Loss
            while epoch[0] != [] and epoch[1] != []:
                batch, lens, epoch = batches_gen.get_batch_from_transfer_domain_epoch(epoch, self.batch_size)
                
                mask = self.getOutputMask(lens)

                mask = Variable(torch.from_numpy(mask)).cuda()
                inputs = Variable(torch.from_numpy(batch)).cuda()
                
                outputs, length = self(inputs.float(), mask, i)
                optimizer.zero_grad()
                loss = None

                if self.loss_type == 'hard_triplet_mmd':
                    loss = self._hard_triplet_mmd(outputs, length)
                    loss.backward()
                elif self.loss_type == 'triplet_mmd':
                    loss = self._triplet_mmd(outputs, length)
                    loss.backward()
                elif self.loss_type == 'norm_triplet_mmd':
                    loss = self._norm_triplet_mmd(outputs, length)
                    loss.backward()
                elif self.loss_type == 'triplet_coral':
                    loss = self._triplet_coral(outputs, length)
                    loss.backward()
                elif self.loss_type == 'quadruplet_loss':
                    loss = self._quadruplet_loss(outputs, length)
                    loss.backward()
                elif self.loss_type == 'icnn_loss':
                    loss = self._icnn_loss(outputs, length)
                    loss.backward()
                
                optimizer.step()

                running_loss += loss.item()
            
                pbar.update(1)

            pbar.close()

            if i >= CHANGE_TRAIN_MODE or (i % 3 == 0 or i > (n_epochs - 3) ):
                for cf in comparison_files:
                    self.new_evaluate(comparison_file=cf, n_epoch=i, result_folder=result_folder)
            
            self.loss_variation.append(running_loss/epoch_size)
            
            lr_scheduler.step()

            torch.save(self.state_dict(), bckp_path + os.sep + "epoch" + str(i) + ".pt")

        # Loss graph
        plt.xlabel("#Epoch")
        plt.ylabel("Loss")
        plt.plot(list(range(0,len(self.loss_variation))), self.loss_variation)
        plt.savefig(result_folder + os.sep + "loss.png")
        plt.cla()
        plt.clf()