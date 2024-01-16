import torch
import torch.nn.functional as F
import torch.nn as nn
from Losses import mmd_loss as mmd
import DTW.soft_dtw_cuda as soft_dtw
import numpy as np
import ds_pipeline

class Triplet_Distillation_MMD(nn.Module):
    def __init__(self, ng : int, nf : int, nw : int, margin : float, alpha : float, beta : float, p : float, r : float, mmd_kernel_num : float, mmd_kernel_mul : float, margin_max : float, margin_min : float):
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
        super(Triplet_Distillation_MMD, self).__init__()
        # Hyperparameters
        self.ng = ng
        self.nf = nf
        self.nw = nw
        self.margin = margin
        self.sdtw = soft_dtw.SoftDTW(True, gamma=5, normalize=False, bandwidth=0.1)
        self.mmd_loss = mmd.MMDLoss(kernel_num=mmd_kernel_num, kernel_mul=mmd_kernel_mul)
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.r = r
        self.margin_max = margin_max
        self.margin_min = margin_min

        self.siz = np.sum(np.array(list(range(1,self.nw+1))))

    def forward(self, data, lens, teacher_data):
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

        for i in range(0, self.nw):
            anchor    = data[i * step]
            positives = data[i * step + 1 : i * step + 1 + self.ng]
            negatives = data[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf]

            len_a = lens[i * step]
            len_p = lens[i * step + 1 : i * step + 1 + self.ng]
            len_n = lens[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf]

            dist_g = torch.zeros((len(positives)), dtype=data.dtype, device=data.device)
            dist_n = torch.zeros((len(negatives)), dtype=data.dtype, device=data.device)

            anchor_t    = teacher_data[i * step]
            positives_t = teacher_data[i * step + 1 : i * step + 1 + self.ng]
            negatives_t = teacher_data[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf]

            dist_g_t = torch.zeros((len(positives)), dtype=teacher_data.dtype, device=teacher_data.device)
            dist_n_t = torch.zeros((len(negatives)), dtype=teacher_data.dtype, device=teacher_data.device)

            '''Average_Pooling_2,4,6'''
            for j in range(len(positives)):
                dist_g[j] = self.sdtw(anchor[None, :int(len_a)], positives[j:j+1, :int(len_p[j])])[0] / (len_a + len_p[j])
                dist_g_t[j] = self.sdtw(anchor_t[None, :int(len_a)], positives_t[j:j+1, :int(len_p[j])])[0] / (len_a + len_p[j])

            for j in range(len(negatives)):
                dist_n[j] = self.sdtw(anchor[None, :int(len_a)], negatives[j:j+1, :int(len_n[j])])[0] / (len_a + len_n[j])
                dist_n_t[j] = self.sdtw(anchor_t[None, :int(len_a)], negatives_t[j:j+1, :int(len_n[j])])[0] / (len_a + len_n[j])

            dists_t = F.relu(-dist_g_t.unsqueeze(0)+dist_n_t.unsqueeze(1))
            fd = ((self.margin_max - self.margin_min)/torch.max(dists_t))*dists_t + self.margin_min

            lk = F.relu((dist_g.unsqueeze(0) - dist_n.unsqueeze(1)) + fd)
            lv = torch.sum(lk) / (lk.data.nonzero(as_tuple=False).size(0) + 1)

            ca = torch.mean(dist_g)
            cb = torch.mean(dist_n[:5])
            intra_loss = torch.sum(dist_g - ca)
            inter_loss = torch.sum(F.relu(self.beta - ((ca - cb).norm(dim=0, p=2))))
            # inter_loss = torch.sum(F.relu(self.beta - torch.abs(ca-cb)))

            user_loss = lv + intra_loss * self.p + inter_loss * self.r

            total_loss += user_loss

        total_loss /= self.nw

        ctr = 0
        mmds = torch.zeros(self.siz, dtype=data.dtype, device=data.device)
        
        for i in range(0, self.nw):
            for j in range(1, self.nw):
                if i != j:
                    mmds[ctr] = self.mmd_loss(data[step*i:step*(i+1) - 5], data[step*j: step*(j+1) - 5]) #* self.alpha
                    ctr+=1

        mmd1 = torch.max(mmds) * self.alpha

        return total_loss + mmd1