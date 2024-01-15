import torch
import torch.nn.functional as F
import torch.nn as nn
from Losses import mmd_loss as mmd
import DTW.soft_dtw_cuda as soft_dtw
import numpy as np

class Hard_Triplet_Loss(nn.Module):
    def __init__(self, ng : int, nf : int, nw : int, margin : float, model_lambda : float):
        """_summary_

        Args:
            ng (torch.nn.Parameter): number of genuine signatures inside the mini-batch
            nf (torch.nn.Parameter): number of forgeries signatures inside the mini-batch
            nw (torch.nn.Parameter): number of mini-batches (writers) inside a batch
            margin (torch.nn.Parameter): triplet loss margin
            model_lambda (torch.nn.Parameter): control intraclass variation
            alpha (torch.nn.Parameter): weighting factor for MMD
            p (torch.nn.Parameter): weighting factor for skilled (p) and random (1-p) forgeries
            q (torch.nn.Parameter): weighting factor for variance of genuines signatures inside the batch
            mmd_kernel_num (torch.nn.Parameter): number of kernels for MMD
            mmd_kernel_mul (torch.nn.Parameter): multipler for MMD
        """
        super(Hard_Triplet_Loss, self).__init__()
        # Hyperparameters
        self.ng = ng
        self.nf = nf
        self.nw = nw
        self.margin = margin
        self.sdtw = soft_dtw.SoftDTW(True, gamma=5, normalize=False, bandwidth=0.1)
        self.model_lambda = model_lambda


        self.siz = np.sum(np.array(list(range(1,self.nw+1))))

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
        n_skilleds = self.nf//2

        for i in range(0, self.nw):
            signatures = data[i * step : i * step + 1 + self.ng + self.nf]

            signatures_lens = lens[i * step : i * step + 1 + self.ng + self.nf]
            
            dists = torch.zeros((len(signatures)), dtype=data.dtype, device=data.device)

            for j in range(len(signatures)):
                dists[j] = self.sdtw(signatures[j:j+1, :int(signatures_lens[j])], signatures[j:j+1, :int(signatures_lens[j])])[0] / (signatures_lens[j] + signatures_lens[j])

            # genuines
            dists_matrix = (dists.unsqueeze(0) + dists.unsqueeze(1)) / 2
            
            # pode ser relevante desconsiderar a distância da âncora com ela mesma
            # aqui ela é considerada
            lks = torch.zeros((self.ng+1,self.ng+1,self.nf), dtype=data.dtype, device=data.device)
            for j in range(0,self.ng+1):
                # lks[j] = F.relu(dists_matrix[j][:self.ng+1].unsqueeze(1) + self.margin - dists_matrix[j][self.ng+1:].unsqueeze(0))
                lks[j] = torch.max(F.relu(dists_matrix[j][:self.ng+1].unsqueeze(1) + self.margin - dists_matrix[j][self.ng+1:].unsqueeze(0)))

            # genuines
            dists_gs_matrix = dists_matrix[:self.ng+1,:self.ng+1]
            mask = torch.tril(torch.ones_like(dists_gs_matrix, dtype=torch.bool), diagonal=-1)
            dists_gs = dists_gs_matrix[mask]

            only_pos = torch.sum(dists_gs) * (self.model_lambda /self.ng)

            lv = torch.sum(lks) / (lks.data.nonzero(as_tuple=False).size(0)+1)

            user_loss = lv + only_pos

            total_loss += user_loss

        total_loss /= self.nw

        return total_loss 