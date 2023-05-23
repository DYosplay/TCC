# import torch
# from torch import nn


# class RBF(nn.Module):

#     def __init__(self, n_kernels=32, mul_factor=2.0, bandwidth=None):
#         super().__init__()
#         self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2).cuda()
#         self.bandwidth = bandwidth

#     def get_bandwidth(self, L2_distances):
#         if self.bandwidth is None:
#             n_samples = L2_distances.shape[0]
#             return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

#         return self.bandwidth

#     def forward(self, X):
#         L2_distances = torch.cdist(X, X) ** 2
#         return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


# class MMDLoss(nn.Module):

#     def __init__(self, kernel=RBF().cuda()):
#         super().__init__()
#         self.kernel = kernel

#     def forward(self, X, Y):
#         K = self.kernel(torch.vstack([X, Y]))

#         X_size = X.shape[0]
#         XX = K[:X_size, :X_size].mean()
#         XY = K[:X_size, X_size:].mean()
#         YY = K[X_size:, X_size:].mean()
#         return XX - 2 * XY + YY

import torch
from torch import nn


class MMDLoss(nn.Module):

    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
    
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        # total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        # total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total.unsqueeze(0)-total.unsqueeze(1))**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss