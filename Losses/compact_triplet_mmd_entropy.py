import torch
import torch.nn.functional as F
import torch.nn as nn
from Losses import mmd_loss as mmd
import DTW.soft_dtw_cuda as soft_dtw
import numpy as np

class Compact_Triplet_MMD_Entropy(nn.Module):
    def __init__(self, ng : int, nf : int, nr : int, nw : int, margin : float, random_margin : float, alpha : float, beta : float, p : float, r : float, mmd_kernel_num : float, mmd_kernel_mul : float):
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
        super(Compact_Triplet_MMD_Entropy, self).__init__()
        # Hyperparameters
        self.ng = ng
        self.nf = nf
        self.nw = nw
        self.nr = nr
        self.margin = margin
        self.random_margin = random_margin
        self.sdtw = soft_dtw.SoftDTW(True, gamma=5, normalize=False, bandwidth=0.1)
        self.mmd_loss = mmd.MMDLoss(kernel_num=mmd_kernel_num, kernel_mul=mmd_kernel_mul)
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.r = r
        self.soft_min = torch.nn.Softmin(dim=0).cuda()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.bce = torch.nn.BCELoss()

        self.siz = np.sum(np.array(list(range(1,self.nw+1))))
        # self.linear = nn.Linear(1, 64, bias=False)

    def forward(self, data, lens,target,n_classes):
        """ Loss de um batch

        Args:
            data (torch tensor): Batch composto por mini self.nw mini-batches. Cada mini-batch é composto por 1+ng+nf (padrão=16) assinaturas e se refere a um único usuário e é (no padrão) disposto da seguinte forma:
            [0]: âncora; [1:ng+1]: amostras positivas; [ng+1:ng+1+nf//2]: falsificações profissionais; [ng+1+nf//2:]: aleatórias 
            lens (torch tensor (int)): tamanho de cada assinatura no batch

        Returns:
            torch.tensor (float): valor da loss
        """
        non_zero_random = 0

        step = (self.ng + self.nf + self.nr+ 1)
        total_loss = 0
        sms = torch.full((self.nw, n_classes), float('inf')).cuda()

        for i in range(0, self.nw):
            anchor    = data[i * step]
            positives = data[i * step + 1 : i * step + 1 + self.ng]
            negatives = data[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf + self.nr]

            len_a = lens[i * step]
            len_p = lens[i * step + 1 : i * step + 1 + self.ng]
            len_n = lens[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf + self.nr]

            # targets = [target[i * step]] + target[i * step + 1 + self.ng + self.nf: i * step + self.ng + self.nf + self.nr + 1]

            dist_g = torch.zeros((len(positives)), dtype=data.dtype, device=data.device)
            dist_n = torch.zeros((len(negatives)), dtype=data.dtype, device=data.device)

            '''Average_Pooling_2,4,6'''
            for j in range(len(positives)):
                dist_g[j] = self.sdtw(anchor[None, :int(len_a)], positives[j:j+1, :int(len_p[j])])[0] / (len_a + len_p[j])

            for j in range(len(negatives)):
                dist_n[j] = self.sdtw(anchor[None, :int(len_a)], negatives[j:j+1, :int(len_n[j])])[0] / (len_a + len_n[j])

            lk_skilled = F.relu(dist_g.unsqueeze(1) + self.margin - dist_n[:self.nf].unsqueeze(0))
            lk_random = F.relu(dist_g.unsqueeze(1) + self.random_margin - dist_n[self.nf:].unsqueeze(0))

            ca = torch.mean(dist_g)
            cb = torch.mean(dist_n[:self.nw])
            intra_loss = torch.sum(dist_g - ca)
            inter_loss = torch.sum(F.relu(self.beta - ((ca - cb).norm(dim=0, p=2))))
            # inter_loss = torch.sum(F.relu(self.beta - torch.abs(ca-cb)))

            lv = (torch.sum(lk_skilled) + torch.sum(lk_random)) / (lk_skilled.data.nonzero(as_tuple=False).size(0) + lk_random.data.nonzero(as_tuple=False).size(0) + 1)

            non_zero_random += lk_random.data.nonzero(as_tuple=False).size(0)

            user_loss = lv + intra_loss * self.p + inter_loss * self.r

            # c_pos_id = torch.argmax(dist_g)
            dist_r   = dist_n[self.nf:]

            labels = torch.tensor([1.0] * self.ng + [0.0] * self.nr).cuda()
            # values = torch.cat((dist_g, dist_r), dim=0)
            # sm = self.soft_min(values)
            # sig = torch.nn.functional.sigmoid(sm)
            # bce = self.bce(sig,labels)

            # user_loss += bce


            bce = 0
            for j in range(0, self.nr):
                sm = self.soft_min(torch.cat((dist_g[0:1], dist_r[j:j+1])))
                # sig = torch.nn.functional.sigmoid(sm)
                bce = bce + self.bce(sm,torch.cat((labels[0:1], labels[self.nr + j: self.nr + j+1])))
            bce = bce / self.nr
            # sm = self.soft_min(values)
            # sig = torch.nn.functional.sigmoid(sm)
            # bce = self.bce(sig,labels)

            user_loss = bce + user_loss


            # labels = torch.full((n_classes,),float('inf'), requires_grad=True).cuda()
            # labels[targets[0]] = torch.max(dist_g)
            # labels[torch.tensor(targets[1:])] = dist_r
            # sm = self.soft_min(labels.double())
            # # sms[i] = sm
            # ground = torch.full((n_classes,),0.0, requires_grad=True).cuda()
            # ground[targets[0]] = 1.0
            # cross_entropy = torch.nn.functional.cross_entropy(sm, ground)
            # cross_entropy = torch.nn.functional.cross_entropy(sm, torch.tensor(targets[0]).cuda())
            # cross_entropy = self.cross_entropy_loss(sm, torch.tensor(targets[0]).cuda())

            

            total_loss += user_loss

        total_loss /= self.nw

        # labels_t = torch.tensor([x for x in range(len(target)) if x % step == 0])
        # target = torch.tensor(target)
        # lab = target[labels_t]
        # cross_entropy = torch.nn.functional.cross_entropy(sms, lab.cuda())

        
        # total_loss += var

        ctr = 0
        mmds = torch.zeros(self.siz, dtype=data.dtype, device=data.device)
        
        for i in range(0, self.nw):
            for j in range(1, self.nw):
                if i != j:
                    mmds[ctr] = self.mmd_loss(data[step*i:step*(i+1) - self.nr], data[step*j: step*(j+1) - self.nr]) #* self.alpha
                    ctr+=1

        mmd1 = torch.max(mmds) * self.alpha

        return total_loss + mmd1, non_zero_random