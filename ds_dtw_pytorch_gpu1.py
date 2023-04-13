from typing import List
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.utils as nutils
from torch.autograd import Variable
import torch.optim as optim
from matplotlib import pyplot as plt
import os
from shapely.geometry import LineString
import math
from tqdm import tqdm
import numpy as np
import batches_gen as batches_gen
import new_soft_dtw
import dtw_cuda
from sklearn.metrics import roc_curve, auc
from typing import Tuple
CHEAT = False
import warnings
warnings.filterwarnings("ignore")
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
# PyTorch models inherit from torch.nn.Module

CUDA0 = torch.device('cuda:0')
#RESULTFOLDER = "RESULT_UNION_10_L2_ALL_FEATURES"

###############################################################################
def lengths_to_mask(lengths, max_len=None, dtype=None):
    """
    Converts a "lengths" tensor to its binary mask representation.
    
    Based on: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397
    
    :lengths: N-dimensional tensor
    :returns: N*max_len dimensional tensor. If max_len==None, max_len=max(lengtsh)
    """
    assert len(lengths.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or lengths.max().item()
    mask = torch.arange(
        max_len,
        device=lengths.device,
        dtype=lengths.dtype)\
    .expand(len(lengths), max_len) < lengths.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=lengths.device)
    return mask


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """
    Masked verstion of the 1D Batch normalization.
    
    Based on: https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py
    
    Receives a N-dim tensor of sequence lengths per batch element
    along with the regular input for masking.
    
    Check pytorch's BatchNorm1d implementation for argument details.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MaskedBatchNorm1d, self).__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats
        )

    def forward(self, inp, lengths):
        self._check_input_dim(inp)

        exponential_average_factor = 0.0
        
        # We transform the mask into a sort of P(inp) with equal probabilities
        # for all unmasked elements of the tensor, and 0 probability for masked
        # ones.
        mask = lengths_to_mask(lengths, max_len=inp.shape[-1], dtype=inp.dtype)
        n = mask.sum()
        mask = mask / n
        mask = mask.unsqueeze(1).expand(inp.shape)

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training and n > 1:
            # Here lies the trick. Using Var(X) = E[X^2] - E[X]^2 as the biased
            # variance, we do not need to make any tensor shape manipulation.
            # mean = E[X] is simply the sum-product of our "probability" mask with the input...
            mean = (mask * inp).sum([0, 2])
            # ...whereas Var(X) is directly derived from the above formulae
            # This should be numerically equivalent to the biased sample variance
            var = (mask * inp ** 2).sum([0, 2]) - mean ** 2
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # Update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        inp = (inp - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
        if self.affine:
            inp = inp * self.weight[None, :, None] + self.bias[None, :, None]

        return inp

###############################################################################

class DsDTW(nn.Module):
    def __init__(self, batch_size : int, in_channels : int, dataset_folder : str, lr : float = 0.01):
        super(DsDTW, self).__init__()

        # Variáveis do modelo
        self.nw = batch_size//16
        self.ng = 5
        self.nf = 10
        self.margin = 1.0
        self.model_lambda = 0.01
        self.lr = lr
        self.n_out = 64
        self.n_hidden = 128
        self.n_in = in_channels
        self.n_layers = 2
        self.batch_size = batch_size
        self.radius = 0

        # Variáveis que lidam com as métricas/resultados
        self.user_err_avg = 0 
        self.dataset_folder = dataset_folder
        self.comparison_data = {}
        self.buffer = "ComparisonFile, mean_local_eer, global_eer, th_global\n"
        self.eer = []
        self.best_eer = math.inf
        self.loss_variation = []
        self.worse = {}

        # Definição da rede
        self.cran  = nn.Sequential(
        nn.Conv1d(in_channels=self.n_in, out_channels=self.n_hidden, kernel_size=4, stride=1, padding=2, bias=True),
        nn.AvgPool1d(4,4, ceil_mode=True),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1)
        )
        # self.bn = MaskedBatchNorm1d(self.n_hidden)

        self.enc1 = torch.nn.TransformerEncoderLayer(self.n_hidden, nhead=1,batch_first=True, dim_feedforward=128, dropout=0.1)
        # self.enc2 = torch.nn.TransformerEncoderLayer(self.n_hidden, nhead=1,batch_first=True, dim_feedforward=128, dropout=0.1)

        # Fecha a update gate (pra virar uma GARU)
        # for i in range(self.n_layers):
        #     eval("self.rnn.bias_hh_l%d"%i)[self.n_hidden:2*self.n_hidden].data.fill_(-1e10) #Initial update gate bias
        #     eval("self.rnn.bias_ih_l%d"%i)[self.n_hidden:2*self.n_hidden].data.fill_(-1e10) #Initial update gate bias
    
        self.linear = nn.Linear(self.n_hidden, 16, bias=False)
        # self.linear2 = nn.Linear(64, 16, bias=False)

        nn.init.kaiming_normal_(self.linear.weight, a=1)
        # nn.init.kaiming_normal_(self.linear2.weight, a=1) 
        nn.init.kaiming_normal_(self.cran[0].weight, a=0)
        # nn.init.kaiming_normal_(self.cran[3].weight, a=0)
        nn.init.zeros_(self.cran[0].bias)
        # nn.init.zeros_(self.cran[3].bias)
        
        # self.new_sdtw_fw = dtw_cuda.DTW(True, normalize=False, bandwidth=0.1)
        self.new_sdtw_fw = new_soft_dtw.SoftDTW(True, gamma=5, normalize=False, bandwidth=1)
        self.new_sdtw = new_soft_dtw.SoftDTW(True, gamma=5, normalize=False, bandwidth=0.1)
        self.dtw = dtw_cuda.DTW(True, normalize=False, bandwidth=1)
        # self.sdtw = soft_dtw_cuda.SoftDTW(True, gamma=5, normalize=False, bandwidth=0.1)

    def getOutputMask(self, lens):    
        lens = np.array(lens, dtype=np.int32)
        lens = (lens+4) //4
        N = len(lens); D = np.max(lens)
        mask = np.zeros((N, D), dtype=np.float32)
        for i in range(N):
            mask[i, 0:lens[i]] = 1.0
        return mask
    
    def forward(self, x, mask):
        length = torch.sum(mask, dim=1)



        h = self.cran(x)
        # h = self.bn(h, length.int())
        
        h = h.transpose(1,2)
        h = h * mask.unsqueeze(2)
        
        # src_mask
        if self.training:
            src_masks = (torch.zeros([self.batch_size, h.shape[1], h.shape[1]], dtype=h.dtype, device=h.device))
            step = (self.ng + self.nf + 1)
            for i in range(0, self.nw):
                anchor = h[i*step]
                for j in range(i*step, (i+1)*step):
                    value, output = self.new_sdtw_fw(anchor[None,], h[j:j+1,])
                    output = output[0][1:h.shape[1]+1, 1:h.shape[1]+1].detach().cpu().numpy()        

                    output = torch.from_numpy(output).cuda()

                    output_mask = (((output - torch.min(output)) / (torch.max(output) - torch.min(output))) + 1)
                    # output_aux = torch.ones(output.shape).cuda()

                    # para a lógica inversa:
                    # output_mask = torch.ones(output.shape).cuda()
                    # output_aux = torch.zeros(output.shape).cuda()

                    # value = 1
                    # output_mask[r, c] = value

                    # # for k in range(1, len(r)):
                    # for k in range(1, self.radius + 1):
                    #     rk_sub = F.relu(r-k).long().cuda()
                    #     ck_sub = F.relu(c-k).long().cuda()
                    #     rk_add = torch.min(c+k, torch.tensor(output.shape[1]-1)).long().cuda()
                    #     ck_add = torch.min(c+k, torch.tensor(output.shape[1]-1)).long().cuda()
                    #     output_mask[rk_sub, ck_sub] = value
                    #     output_mask[rk_sub, c]      = value
                    #     output_mask[rk_sub, ck_add] = value

                    #     output_mask[rk_add, ck_sub] = value
                    #     output_mask[rk_add, c]      = value
                    #     output_mask[rk_add, ck_add] = value

                    #     output_mask[r, ck_add]      = value
                    #     output_mask[r, ck_sub]      = value

                    src_masks[j] = output_mask
            
            h = self.enc1(src=h, src_mask=src_masks, src_key_padding_mask=(~mask.bool()))
            # h = self.enc2(src=h, src_key_padding_mask=(~mask.bool()))
        else:
            src_masks = torch.zeros([h.shape[0], h.shape[1], h.shape[1]], dtype=h.dtype, device=h.device)
            sign = h[0]

            for i in range(len(h)):
                value, output = self.new_sdtw_fw(sign[None, ], h[i:i+1, ])
                output = output[0][1:h.shape[1]+1, 1:h.shape[1]+1]

                output_mask = (((output - torch.min(output)) / (torch.max(output) - torch.min(output))) + 1)
                # output_aux = torch.ones(output.shape).cuda()

                # para a lógica inversa:
                # output_mask = torch.ones(output.shape).cuda()
                # output_aux = torch.zeros(output.shape).cuda()

                # value = 1
                # output_mask[r, c] = value

                # for k in range(1, self.radius + 1):
                #     rk_sub = F.relu(r-k).long().cuda()
                #     ck_sub = F.relu(c-k).long().cuda()
                #     rk_add = torch.min(c+k, torch.tensor(output.shape[1]-1)).long().cuda()
                #     ck_add = torch.min(c+k, torch.tensor(output.shape[1]-1)).long().cuda()
                #     output_mask[rk_sub, ck_sub] = value
                #     output_mask[rk_sub, c]      = value
                #     output_mask[rk_sub, ck_add] = value

                #     output_mask[rk_add, ck_sub] = value
                #     output_mask[rk_add, c]      = value
                #     output_mask[rk_add, ck_add] = value

                #     output_mask[r, ck_add]      = value
                #     output_mask[r, ck_sub]      = value

                src_masks[i] = output_mask
            
            h = self.enc1(src=h, src_mask=src_masks, src_key_padding_mask=(~mask.bool()))
            # h = self.enc2(src=h, src_key_padding_mask=(~mask.bool()))

        h = self.linear(h)

        if self.training:
            return F.avg_pool1d(h.permute(0,2,1),2,2,ceil_mode=False).permute(0,2,1), (length//2).float()

        return h * mask.unsqueeze(2), length.float()

    def _traceback(self, acc_cost_matrix ):
        """ Encontra a path (matching) dado uma matriz de custo acumulado obtida a partir do cálculo do DTW.
        Args:
            acc_cost_matrix (npt.DTypeLike): matriz de custo acumulado obtida a partir do cálculo do DTW.

        Returns:
            Tuple[npt.ArrayLike, npt.ArrayLike]: coordenadas ponto a ponto referente a primeira e segunda sequência utilizada no cálculo do DTW.
        """
        # acc_cost_matrix = np.transpose(acc_cost_matrix)
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


    def loss(self, data, lens):
        """ 
            Loss de um batch
        """
        step = (self.ng + self.nf + 1)
        l    = 0
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

            # aa = self.new_sdtw(anchor[None, :int(len_a)], anchor[None, :int(len_a)])
            '''Average_Pooling_2,4,6'''
            for i in range(len(positives)):
                dist_g[i] = self.new_sdtw(anchor[None, :int(len_a)], positives[i:i+1, :int(len_p[i])])[0] / (len_a + len_p[i])
                
                # ap = self.new_sdtw(anchor[None, :int(len_a)], positives[i:i+1, :int(len_p[i])])
                # pp = self.new_sdtw(positives[i:i+1, :int(len_p[i])], positives[i:i+1, :int(len_p[i])])

                # dist_g[i] = (ap - (0.5 * (aa+pp))) / (len_a + len_p[i])

            for i in range(len(negatives)):
                dist_n[i] = self.new_sdtw(anchor[None, :int(len_a)], negatives[i:i+1, :int(len_n[i])])[0] / (len_a + len_n[i])
                
                # an = self.new_sdtw(anchor[None, :int(len_a)], negatives[i:i+1, :int(len_n[i])])
                # nn = self.new_sdtw(negatives[i:i+1, :int(len_n[i])], negatives[i:i+1, :int(len_n[i])])

                # dist_n[i] = (an - (0.5 * (aa+nn))) / (len_a + len_n[i])

            only_pos = torch.sum(dist_g) * (self.model_lambda /self.ng)
            
            lk = 0
            non_zeros = 1
            for g in dist_g:
                for n in dist_n:
                    temp = F.relu(g + self.margin - n)
                    if temp > 0:
                        lk += temp
                        non_zeros+=1

            lk /= non_zeros

            user_loss = lk + only_pos

            total_loss += user_loss
        
        total_loss /= self.nw

        return total_loss

    def dte(self, x, y, len_x, len_y):
        #3 usando dtw cuda
        return self.dtw(x[None, :int(len_x)], y[None, :int(len_y)])[0] /(64* (len_x + len_y))
        # return self.dtw(x[None, :int(len_x)], y[None, :int(len_y)]) /((len_x + len_y))
        
        #usando fastdtw
        # d, _ = fastdtw(x[:int(len_x)].cpu().detach().numpy(), y[:int(len_y)].cpu().detach().numpy(), dist=2, radius=2)
        # return torch.tensor([d / ((len_x + len_y))])
        
        # return self.dtw(x[None, :int(len_x)], y[None, :int(len_y)]) /((len_x + len_y))
        # d, _ = dtw(x[:int(len_x)].cpu().detach().numpy(), y[:int(len_y)].cpu().detach().numpy(), dist=2, radius=10)
        # return torch.tensor([d / (len_x + len_y)])
        
        #d,_,_,_ = my_dtw.my_dtw(x.cpu().detach().numpy(), y.cpu().detach().numpy(), dist='cityblock')
        

    def _generate_graph(self, legit : List[float], forgery : List[float], epoch, result_folder : str, user : str = '0'):
        total_distances = np.array(legit + forgery)
        total_distances = np.sort(total_distances)

        frr_list = []
        far_list = []

        for dist in total_distances:
            frr = np.sum(legit >= dist) / len(legit)
            frr_list.append(frr)
            
            far = np.sum(forgery < dist) / len(forgery)
            far_list.append(far)

        frr_list = np.array(frr_list)
        far_list = np.array(far_list)

        if not os.path.exists(result_folder + os.sep + user):
            os.mkdir(result_folder + os.sep + user)

        plt.plot(total_distances, frr_list, 'r', label="FRR")
        plt.plot(total_distances, far_list, 'b', label="FAR")
        plt.legend(loc="upper right")

        line_1 = LineString(np.column_stack((total_distances, frr_list)))
        line_2 = LineString(np.column_stack((total_distances, far_list)))
        intersection = line_1.intersection(line_2)
        x,y = intersection.xy
        
        plt.xlabel("Threshold")
        plt.ylabel("Error Rate")
        plt.plot(*intersection.xy, 'ro')
        plt.text(x[0]+0.05,y[0]+0.05, "EER = " + "{:.3f}".format(y[0]))
        plt.savefig(result_folder + os.sep + user + os.sep + "Epoch" + str(epoch) + ".png")
        plt.cla()
        plt.clf()

        return y[0], x[0]

    def start_train(self, n_epochs : int, batch_size : int, comparison_files : List[str], result_folder : str):
        
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) 

        running_loss = 0

        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        if not os.path.exists(result_folder + os.sep + "Backup"):
            os.mkdir(result_folder + os.sep + "Backup")
        bckp_path = result_folder + os.sep + "Backup"

        for i in range(1, n_epochs+1):
            epoch = batches_gen.generate_epoch()
            epoch_size = len(epoch)
            pbar = tqdm(total=(epoch_size//(batch_size//16)), position=0, leave=True, desc="Epoch " + str(i) +" PAL: " + "{:.2f}".format(running_loss/epoch_size))

            running_loss = 0
            
            #PAL = Previous Accumulated Loss
            while epoch != []:
                batch, lens, epoch = batches_gen.get_batch_from_epoch(epoch, batch_size)
                
                mask = self.getOutputMask(lens)
                mask = Variable(torch.from_numpy(mask)).cuda()
                inputs = Variable(torch.from_numpy(batch)).cuda()
                
                optimizer.zero_grad()
                outputs, length = self(inputs.float(), mask)

                loss = self.loss(outputs, length)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
                pbar.update(1)

            pbar.close()
          
            # if i % 5 == 0: self.new_evaluate(comparison_file=comparison_files[0], n_epoch=i, result_folder=result_folder)
            if i % 5 == 0 or i > (n_epochs - 3): 
                for cf in comparison_files:
                    # self.evaluate(comparions_files=comparison_files, n_epoch=i, result_folder=result_folder)
                    self.new_evaluate(comparison_file=cf, n_epoch=i, result_folder=result_folder)
              #  self.margin -= 0.5
            
            self.loss_variation.append(running_loss/epoch_size)
            
            lr_scheduler.step()

            torch.save(self.state_dict(), bckp_path + os.sep + "epoch" + str(i) + ".pt")

        # Loss graph
        plt.xlabel("#Epoch")
        plt.ylabel("Loss")
        plt.plot(list(range(0,n_epochs)), self.loss_variation)
        plt.savefig(result_folder + os.sep + "loss.png")
        plt.cla()
        plt.clf()

    def inference(self, files : str) -> Tuple[float, str, int]:
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

        test_batch, lens = batches_gen.files2array(refs + [sign], developtment=CHEAT)

        mask = self.getOutputMask(lens)
        
        mask = Variable(torch.from_numpy(mask)).cuda()
        inputs = Variable(torch.from_numpy(test_batch)).cuda()

        embeddings, lengths = self(inputs.float(), mask)    
        refs = embeddings[:len(embeddings)-1]
        sign = embeddings[-1]

        len_refs = lengths[:len(embeddings)-1]
        len_sign = lengths[-1]

        dk = math.nan
        count = 0
        if len(refs) == 1 : dk = 1
        else:
            dk = 0
            for i in range(0, len(refs)):
                for j in range(1, len(refs)):
                    if i < j:
                        dk += self.dte(refs[i], refs[j], len_refs[i], len_refs[j])
                        count += 1

            dk = dk/(count)
    
        dk_sqrt = math.sqrt(dk)
        
        dists = []
        for i in range(0, len(refs)):
            dists.append(self.dte(refs[i], sign, len_refs[i], len_sign).detach().cpu().numpy()[0])

        dists = np.array(dists) / dk_sqrt

        s_avg = np.mean(dists)
        s_min = min(dists)

        return s_avg + s_min, user_key, result

    def get_eer(self, y_true = List[int], y_scores = List[int], result_folder : str = None, generate_graph : bool = False, n_epoch : int = None) -> Tuple[float, float]:
        fpr, tpr, threshold = roc_curve(y_true=y_true, y_score=y_scores, pos_label=1)
        fnr = 1 - tpr

        far = fpr
        frr = fnr

        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # as a sanity check the value should be close to
        eer2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

        eer = (eer + eer2)/2

        if generate_graph:
            frr_list = np.array(frr)
            far_list = np.array(far)

            plt.plot(threshold, frr_list, 'r', label="FRR")
            plt.plot(threshold, far_list, 'b', label="FAR")
            plt.legend(loc="upper right")
 
            plt.xlabel("Threshold")
            plt.ylabel("Error Rate")
            plt.plot(eer_threshold, eer, 'ro')
            plt.text(eer_threshold + 0.05, eer+0.05, s="EER = " + "{:.5f}".format(eer))
            #plt.text(eer_threshold + 1.05, eer2+1.05, s="EER = " + "{:.5f}".format(eer2))
            plt.savefig(result_folder + os.sep + "Epoch" + str(n_epoch) + ".png")
            plt.cla()
            plt.clf()

        return eer, eer_threshold

    def new_evaluate(self, comparison_file : str, n_epoch : int, result_folder : str):
        # Para cada usuário gerar duas listas: uma com as labels e a outra com a predição
        # 0 é positivo (original), 1 é negativo (falsificação)
        self.train(mode=False)
        lines = []
        with open(comparison_file, "r") as fr:
            lines = fr.readlines()

        file_name = (comparison_file.split(".")[0]).split(os.sep)[-1]
        print("\n\tAvaliando " + file_name)
        comparison_folder = result_folder + os.sep + file_name
        if not os.path.exists(comparison_folder): os.mkdir(comparison_folder)

        users = {}

        for line in tqdm(lines, "Calculando distâncias..."):
            distance, user_id, true_label = self.inference(line)
            
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

            eer, eer_threshold = self.get_eer(y_true=users[user]["true_label"], y_scores=users[user]["distances"])
            eers.append(eer)
            local_buffer += user + ", " + "{:.5f}".format(eer) + ", " + "{:.5f}".format(eer_threshold) + "\n"

        print("Obtendo EER global...")
        
        # Calculo do EER global
        eer_global, eer_threshold_global = self.get_eer(global_true_label, global_distances, result_folder=comparison_folder, generate_graph=True, n_epoch=n_epoch)

        buffer += "Global, " + "{:.5f}".format(eer_global) + ", " + "{:.5f}".format(eer_threshold_global) + "\n" + local_buffer

        local_eer_mean = np.mean(np.array(eers))
        self.buffer += file_name + ", " + "{:.5f}".format(local_eer_mean) + ", " + "{:.5f}".format(eer_global) + ", " + "{:.5f}".format(eer_threshold_global) + "\n"

        with open(comparison_folder + os.sep + file_name + " epoch=" + str(n_epoch) + ".csv", "w") as fw:
            fw.write(buffer)

        if eer_global < self.best_eer:
            torch.save(self.state_dict(), result_folder + os.sep + "Backup" + os.sep + "best.pt")
        
        self.train(mode=True)
