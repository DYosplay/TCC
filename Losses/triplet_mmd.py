import torch
import torch.nn.functional as F
import torch.nn as nn
from Losses import mmd_loss as mmd
import DTW.soft_dtw_cuda as soft_dtw

class Triplet_MMD(nn.Module):
    def __init__(self, ng : torch.nn.Parameter, nf : torch.nn.Parameter, nw : torch.nn.Parameter, margin : torch.nn.Parameter, model_lambda : torch.nn.Parameter, alpha : torch.nn.Parameter, p : torch.nn.Parameter, r : torch.nn.Parameter, mmd_kernel_num : torch.nn.Parameter, mmd_kernel_mul : torch.nn.Parameter):
        """_summary_

        Args:
            ng (torch.nn.Parameter): number of genuine signatures inside the mini-batch
            nf (torch.nn.Parameter): number of forgeries signatures inside the mini-batch
            nw (torch.nn.Parameter): number of mini-batches (writers) inside a batch
            margin (torch.nn.Parameter): triplet loss margin
            model_lambda (torch.nn.Parameter): control intraclass variation
            alpha (torch.nn.Parameter): weighting factor for MMD
            p (torch.nn.Parameter): weighting factor for skilled (p) and random (1-p) forgeries
            r (torch.nn.Parameter): weighting factor for variance of genuines signatures inside the batch
            mmd_kernel_num (torch.nn.Parameter): number of kernels for MMD
            mmd_kernel_mul (torch.nn.Parameter): multipler for MMD
        """
        super(Triplet_MMD, self).__init__()
        self.ng = ng
        self.nf = nf
        self.nw = nw
        self.model_lambda = model_lambda
        self.margin = margin
        self.sdtw = soft_dtw.SoftDTW(True, gamma=5, normalize=False, bandwidth=0.1)
        self.mmd_loss = mmd.MMDLoss(kernel_num=mmd_kernel_num, kernel_mul=mmd_kernel_mul)
        self.alpha = alpha
        self.p = p
        self.r = r

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

            only_pos = torch.sum(dist_g) * (self.model_lambda /self.ng)

            lk1 = F.relu(dist_g.unsqueeze(1) + self.margin - dist_n[:5].unsqueeze(0)) * self.p
            lk2 = F.relu(dist_g.unsqueeze(1) + self.margin - dist_n[5:].unsqueeze(0)) * (1-self.p)
            lv = (torch.sum(lk1) + torch.sum(lk2)) / (lk1.data.nonzero(as_tuple=False).size(0) + lk2.data.nonzero(as_tuple=False).size(0) + 1)

            user_loss = lv + only_pos

            total_loss += user_loss

        total_loss /= self.nw
        
        mmd1 = self.mmd_loss(data[0:step - 5], data[step: step*2 - 5]) * self.alpha
        var_g = torch.var(dists_gs) * self.r

        return total_loss + mmd1 + var_g 