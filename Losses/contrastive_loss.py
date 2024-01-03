import torch
import torch.nn.functional as F
import torch.nn as nn
from Losses import mmd_loss as mmd
import DTW.soft_dtw_cuda as soft_dtw

class Contrastive_Loss(nn.Module):
    def __init__(self, ng : torch.nn.Parameter, nf : torch.nn.Parameter, nw : torch.nn.Parameter, margin : torch.nn.Parameter):
        """_summary_

        Args:
            ng (torch.nn.Parameter): number of genuine signatures inside the mini-batch
            nf (torch.nn.Parameter): number of forgeries signatures inside the mini-batch
            nw (torch.nn.Parameter): number of mini-batches (writers) inside a batch
            margin (torch.nn.Parameter): triplet loss margin
        """
        super(Contrastive_Loss, self).__init__()
        self.ng = ng
        self.nf = nf
        self.nw = nw
        self.margin = margin
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

            dist_aa = self.sdtw(anchor[None, :int(len_a)], anchor[None, :int(len_a)])[0] / (len_a + len_a)

            '''Average_Pooling_2,4,6'''
            for j in range(len(positives)): 
                dist_g[j] = self.sdtw(anchor[None, :int(len_a)], positives[j:j+1, :int(len_p[j])])[0] / (len_a + len_p[j])
                dists_gs[i*self.ng + j] = dist_g[j]

            for j in range(len(negatives)):
                dist_n[j] = self.sdtw(anchor[None, :int(len_a)], negatives[j:j+1, :int(len_n[j])])[0] / (len_a + len_n[j])
                dists_ns[i*self.nf + j] = dist_n[j]

            lk1 = (dist_g - dist_aa)
            lk2 = F.relu(self.margin - (dist_n - dist_aa))

            # lk1 = F.relu(dist_g.unsqueeze(1) + self.margin - dist_n[:5].unsqueeze(0))
            # lk2 = F.relu(dist_g.unsqueeze(1) + self.margin - dist_n[5:].unsqueeze(0))
            lv = (torch.sum(lk1) + torch.sum(lk2)) / (lk1.data.nonzero(as_tuple=False).size(0) + lk2.data.nonzero(as_tuple=False).size(0) + 1)

            total_loss += lv

        total_loss /= self.nw
        
        # mmd1 = self.mmd_loss(data[0:step - 5], data[step: step*2 - 5]) * self.alpha
        # var_g = torch.var(dists_gs) * self.r

        return total_loss #+ mmd1 + var_g 