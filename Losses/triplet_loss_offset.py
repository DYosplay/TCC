import torch
import torch.nn.functional as F
import torch.nn as nn
from Losses import mmd_loss as mmd
import DTW.soft_dtw_cuda as soft_dtw
import numpy as np

class Triplet_Loss_Offset(nn.Module):
    def __init__(self, ng : torch.nn.Parameter, nf : torch.nn.Parameter, nw : torch.nn.Parameter, margin : torch.nn.Parameter, model_lambda : torch.nn.Parameter, alpha : torch.nn.Parameter):
        """_summary_

        Args:
            ng (torch.nn.Parameter): number of genuine signatures inside the mini-batch
            nf (torch.nn.Parameter): number of forgeries signatures inside the mini-batch
            nw (torch.nn.Parameter): number of mini-batches (writers) inside a batch
            margin (torch.nn.Parameter): triplet loss margin
            model_lambda (torch.nn.Parameter): control intraclass variation

        """
        super(Triplet_Loss_Offset, self).__init__()
        # Hyperparameters
        self.ng = ng
        self.nf = nf
        self.nw = nw
        self.model_lambda = model_lambda
        self.margin = margin
        self.alpha = alpha
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

            # lk = torch.sum(F.relu(dist_g.unsqueeze(1) + self.margin - dist_n.unsqueeze(0)))
            # lv = lk / (lk.data.nonzero(as_tuple=False).size(0) + 1)

            median_g = torch.median(dist_g)
            median_s = torch.median(dist_n[:5])
            
            gs = dist_g < median_g
            ss = dist_n[:5] < median_g
            err1 = torch.sum(torch.bitwise_and(gs,ss))
            gs = dist_g > median_s
            ss = dist_n[:5] > median_s
            err1 += torch.sum(torch.bitwise_and(gs,ss))

            rs = dist_n[5:] < median_s

            gs = dist_g < median_g
            rs = dist_n[5:] < median_g
            err2 = torch.sum(torch.bitwise_and(gs,rs))
            gs = dist_g > median_s
            rs = dist_n[5:] > median_s
            err2 += torch.sum(torch.bitwise_and(gs,rs))

            offset = (err1 + err2) / (self.nf * self.ng)
            # if offset > 0:
            #     print("offset: " +  str(offset))


            user_loss = lv + only_pos + offset * self.alpha

            total_loss += user_loss
    
        total_loss /= self.nw
        triplet_loss = total_loss

        return triplet_loss