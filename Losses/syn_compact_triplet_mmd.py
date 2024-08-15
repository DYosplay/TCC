import torch
import torch.nn.functional as F
import torch.nn as nn
from Losses import mmd_loss as mmd
import DTW.soft_dtw_cuda as soft_dtw
import numpy as np

class Syn_Compact_Triplet_MMD(nn.Module):
    def __init__(self, ng : int, ns : int, nr : int, sng : int, sns : int, nw : int, margin : float, alpha : float, beta : float, p : float, r : float, mmd_kernel_num : float, mmd_kernel_mul : float):
        """_summary_

        Args:
            ng (torch.nn.Parameter): number of genuine signatures inside the mini-batch
            ns (torch.nn.Parameter): number of skilled forgeries signatures inside the mini-batch
            nr (torch.nn.Parameter): number of random forgeries signatures inside the mini-batch
            sng (torch.nn.Parameter): number of synthetic genuine signatures inside the mini-batch
            sns (torch.nn.Parameter): number of synthetic skilled signatures inside the mini-batch
            nw (torch.nn.Parameter): number of writers inside batch
            margin (torch.nn.Parameter): triplet loss margin
            model_lambda (torch.nn.Parameter): control intraclass variation
            alpha (torch.nn.Parameter): weighting factor for MMD
            p (torch.nn.Parameter): weighting factor for skilled (p) and random (1-p) forgeries
            q (torch.nn.Parameter): weighting factor for variance of genuines signatures inside the batch
            mmd_kernel_num (torch.nn.Parameter): number of kernels for MMD
            mmd_kernel_mul (torch.nn.Parameter): multipler for MMD
        """
        super(Syn_Compact_Triplet_MMD, self).__init__()
        # Hyperparameters
        self.ng = ng
        self.ns = ns
        self.nr = nr
        self.nw = nw
        self.sng = sng
        self.sns = sns
        self.margin = margin
        self.sdtw = soft_dtw.SoftDTW(True, gamma=5, normalize=False, bandwidth=0.1)
        self.mmd_loss = mmd.MMDLoss(kernel_num=mmd_kernel_num, kernel_mul=mmd_kernel_mul)
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.r = r


        self.siz = np.sum(np.array(list(range(1,self.nw+1))))

    def forward(self, data, lens):
        """ Loss de um batch

        Args:
            data (torch tensor): Batch composto por mini self.nw mini-batches. Cada mini-batch é composto por 1+ng+sng+ns+sns+nr (padrão=16) assinaturas
            lens (torch tensor (int)): tamanho de cada assinatura no batch

        Returns:
            torch.tensor (float): valor da loss
        """

        step = (self.ng + self.sng + self.ns + self.sns + self.nr + 1)
        total_loss = 0

        for i in range(0, self.nw):
            anchor    = data[i * step]
            positives = data[i * step + 1 : i * step + 1 + self.ng]
            syn_positives = data[i * step + 1 + self.ng : i * step + 1 + self.ng + self.sng]
            negatives = data[i * step + 1 + self.ng + self.sng :i * step + 1 + self.ng + self.sng + self.ns]
            syn_negatives = data[i * step + 1 + self.ng + self.sng + self.ns:i * step + 1 + self.ng + self.sng + self.ns + self.sns]
            ran_negatives = data[i * step + 1 + self.ng + self.sng + self.ns + self.sns:i * step + 1 + self.ng + self.sng + self.ns + self.sns + self.nr]

            len_a = data[i * step]
            len_p = data[i * step + 1 : i * step + 1 + self.ng]
            len_sg = data[i * step + 1 + self.ng : i * step + 1 + self.ng + self.sng]
            len_n = data[i * step + 1 + self.ng + self.sng :i * step + 1 + self.ng + self.sng + self.ns]
            len_sn = data[i * step + 1 + self.ng + self.sng + self.ns:i * step + 1 + self.ng + self.sng + self.ns + self.sns]
            len_rn = data[i * step + 1 + self.ng + self.sng + self.ns + self.sns:i * step + 1 + self.ng + self.sng + self.ns + self.sns + self.nr]

            dist_p = torch.zeros((len(positives)), dtype=data.dtype, device=data.device)
            dist_sp = torch.zeros((len(syn_positives)), dtype=data.dtype, device=data.device)
            dist_s = torch.zeros((len(negatives)), dtype=data.dtype, device=data.device)
            dist_ss = torch.zeros((len(syn_negatives)), dtype=data.dtype, device=data.device)
            dist_r = torch.zeros((len(ran_negatives)), dtype=data.dtype, device=data.device)

            '''Average_Pooling_2,4,6'''
            for j in range(len(positives)):
                dist_p[j] = self.sdtw(anchor[None, :int(len_a)], positives[j:j+1, :int(len_p[j])])[0] / (len_a + len_p[j])

            for j in range(len(syn_positives)):
                dist_sp[j] = self.sdtw(anchor[None, :int(len_a)], syn_positives[j:j+1, :int(len_sg[j])])[0] / (len_a + len_sg[j])

            for j in range(len(negatives)):
                dist_s[j] = self.sdtw(anchor[None, :int(len_a)], negatives[j:j+1, :int(len_n[j])])[0] / (len_a + len_n[j])

            for j in range(len(syn_negatives)):
                dist_ss[j] = self.sdtw(anchor[None, :int(len_a)], syn_negatives[j:j+1, :int(len_sn[j])])[0] / (len_a + len_sn[j])
            
            for j in range(len(ran_negatives)):
                dist_r[j] = self.sdtw(anchor[None, :int(len_a)], ran_negatives[j:j+1, :int(len_rn[j])])[0] / (len_a + len_rn[j])

            dist_g = torch.cat((dist_p, dist_sp), dim=0)
            dist_n = torch.cat((dist_s, dist_ss, dist_r), dim=0)


            lk = F.relu(dist_g.unsqueeze(1) + self.margin - dist_n.unsqueeze(0))

            ca = torch.mean(dist_g)
            cb = torch.mean(dist_n[:5])
            intra_loss = torch.sum(dist_g - ca)
            inter_loss = torch.sum(F.relu(self.beta - ((ca - cb).norm(dim=0, p=2))))
            # inter_loss = torch.sum(F.relu(self.beta - torch.abs(ca-cb)))

            lv = torch.sum(lk) / (lk.data.nonzero(as_tuple=False).size(0) + 1)

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