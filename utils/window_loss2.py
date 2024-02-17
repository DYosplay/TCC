import torch
import torch.nn.functional as F
import torch.nn as nn
from Losses import mmd_loss as mmd
import DTW.soft_dtw_cuda as soft_dtw
import numpy as np

class Window_Loss2(nn.Module):
    def __init__(self, ng : torch.nn.Parameter, nf : torch.nn.Parameter, nw : torch.nn.Parameter, margin : torch.nn.Parameter, model_lambda : torch.nn.Parameter, r : float, p : float):
        """_summary_

        Args:
            ng (torch.nn.Parameter): number of genuine signatures inside the mini-batch
            nf (torch.nn.Parameter): number of forgeries signatures inside the mini-batch
            nw (torch.nn.Parameter): number of mini-batches (writers) inside a batch
            margin (torch.nn.Parameter): triplet loss margin
            model_lambda (torch.nn.Parameter): control intraclass variation

        """
        super(Window_Loss2, self).__init__()
        # Hyperparameters
        self.ng = ng
        self.nf = nf
        self.nw = nw
        self.model_lambda = model_lambda
        self.margin = margin
        self.r = r
        self.p = p
        self.sdtw = soft_dtw.SoftDTW(True, gamma=5, normalize=False, bandwidth=0.1)

    def forward(self, data, lens):
        """ Loss de um batch

        Args:
            data (torch tensor): Batch composto por mini self.nw mini-batches. Cada mini-batch é composto por 1+ng+nf (padrão=16) assinaturas e se refere a um único usuário e é (no padrão) disposto da seguinte forma:
            [0]: âncora; [1:ng+1]: amostras positivas; [ng+1:ng+1+nf//2]: falsificações profissionais; [ng+1+nf//2:]: aleatórias 
            lens (torch tensor (int)): tamanho de cada assinatura no batch

        Returns:
            torch.tensor (float): valor da loss
        """

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

            lk1 = F.relu(dist_g.unsqueeze(1) + self.margin - dist_n[:5].unsqueeze(0)) * self.p
            lk2 = F.relu(dist_g.unsqueeze(1) + self.margin - dist_n[5:].unsqueeze(0)) * (1-self.p)
            lv = (torch.sum(lk1) + torch.sum(lk2)) / (lk1.data.nonzero(as_tuple=False).size(0) + lk2.data.nonzero(as_tuple=False).size(0) + 1)
            total_loss += lv

        total_loss /= self.nw

        smooth = self.r

        expg = torch.exp(smooth * dists_gs)
        dp = torch.sum(dists_gs * expg)/torch.sum(expg)

        expn = torch.exp(-smooth * dists_ns)
        dn = torch.sum(dists_ns * expn)/torch.sum(expn)

        window_loss = F.relu(dp - dn)

        return (window_loss * self.p) + (total_loss * (1 - self.p))