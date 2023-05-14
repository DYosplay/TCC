from typing import List, Tuple
from numpy import typing as npt

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.optim as optim

from matplotlib import pyplot as plt
import os
import math
from tqdm import tqdm
import numpy as np

import DataLoader.batches_gen as batches_gen
import DTW.soft_dtw_cuda as soft_dtw
import DTW.dtw_cuda as dtw
import utils.metrics as metrics

CHEAT = False
CHANGE_TRAIN_MODE=20

import warnings
warnings.filterwarnings("ignore")

class DsTransformer(nn.Module):
    def __init__(self, batch_size : int, in_channels : int, dataset_folder : str, gamma : int, lr : float = 0.01):
        super(DsTransformer, self).__init__()

        # Variáveis do modelo
        self.nw = batch_size//16
        self.ng = 5
        self.nf = 10
        self.margin = 1
        self.model_lambda = 0.01
        self.lr = lr
        self.n_out = 64
        self.n_hidden = 128
        self.n_in = in_channels
        self.batch_size = batch_size
        self.gamma = gamma

        # variáveis para a loss
        self.scores = []
        self.labels = []
        self.th = 3.5
        # self.loss_value = math.inf


        # Variáveis que lidam com as métricas/resultados
        self.user_err_avg = 0 
        self.dataset_folder = dataset_folder
        self.comparison_data = {}
        self.buffer = "ComparisonFile, mean_local_eer, global_eer, th_global\n"
        self.eer = []
        self.best_eer = math.inf
        self.loss_variation = []

        # Definição da rede
        self.cran  = nn.Sequential(
        nn.Conv1d(in_channels=self.n_in, out_channels=self.n_hidden, kernel_size=4, stride=1, padding=2, bias=True),
        nn.AvgPool1d(4,4, ceil_mode=True),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1)
        )

        nn.init.kaiming_normal_(self.cran[0].weight, a=0)
        nn.init.zeros_(self.cran[0].bias)

        self.enc1 = torch.nn.TransformerEncoderLayer(self.n_hidden, nhead=1,batch_first=True, dim_feedforward=128, dropout=0.1)
        
        self.linear = nn.Linear(self.n_hidden, 16, bias=False)
        nn.init.kaiming_normal_(self.linear.weight, a=1)
        
        self.sdtw = soft_dtw.SoftDTW(True, gamma=5, normalize=False, bandwidth=0.1)
        self.dtw = dtw.DTW(True, normalize=False, bandwidth=1)

        # Segunda parte

        # self.cran2  = nn.Sequential(
        # nn.Conv1d(in_channels=16, out_channels=self.n_hidden, kernel_size=4, stride=1, padding=2, bias=True),
        # nn.ReLU(inplace=True),
        # nn.Dropout(0.1))
        # nn.init.kaiming_normal_(self.cran2[0].weight, a=0)
        # nn.init.zeros_(self.cran2[0].bias)

        self.enc2 = torch.nn.TransformerEncoderLayer(self.n_hidden, nhead=1,batch_first=True, dim_feedforward=128, dropout=0.1)

        self.linear2 = nn.Linear(16, self.n_hidden, bias=False)
        self.linear3 = nn.Linear(self.n_hidden, 16, bias=False)
        nn.init.kaiming_normal_(self.linear.weight, a=1)
        

    def getOutputMask(self, lens):    
        lens = np.array(lens, dtype=np.int32)
        lens = (lens+4) //4
        N = len(lens); D = np.max(lens)
        mask = np.zeros((N, D), dtype=np.float32)
        for i in range(N):
            mask[i, 0:lens[i]] = 1.0
        return mask

    def forward(self, x, mask, n_epoch):
        length = torch.sum(mask, dim=1)

        h = self.cran(x)
        
        h = h.transpose(1,2)
        h = h * mask.unsqueeze(2)
        
        # src_mask
        if self.training:
            src_masks = (torch.zeros([self.batch_size, h.shape[1], h.shape[1]], dtype=h.dtype, device=h.device))
            
            # step = (self.ng + self.nf + 1)
            # for i in range(0, self.nw):
            #     anchor = h[i*step]
            #     for j in range(i*step, (i+1)*step):
            #         value, output = self.dtw(anchor[None,], h[j:j+1,])
            #         output = output[0][1:h.shape[1]+1, 1:h.shape[1]+1].detach().cpu().numpy()        
            #         output = torch.from_numpy(output).cuda()
            #         output_mask = (((output - torch.min(output)) / (torch.max(output) - torch.min(output))) + 1)
            #         src_masks[j] = output_mask
            
            h = self.enc1(src=h, src_mask=src_masks, src_key_padding_mask=(~mask.bool()))

        else:
            src_masks = torch.zeros([h.shape[0], h.shape[1], h.shape[1]], dtype=h.dtype, device=h.device)
            
            # sign = h[0]
            # for i in range(len(h)):
            #     value, output = self.dtw(sign[None, ], h[i:i+1, ])
            #     output = output[0][1:h.shape[1]+1, 1:h.shape[1]+1].detach().cpu().numpy()        
            #     output = torch.from_numpy(output).cuda()
            #     output_mask = (((output - torch.min(output)) / (torch.max(output) - torch.min(output))) + 1)
            #     src_masks[i] = output_mask
            
            h = self.enc1(src=h, src_mask=src_masks, src_key_padding_mask=(~mask.bool()))

        h = self.linear(h)

        if n_epoch <= CHANGE_TRAIN_MODE:

            if self.training:
                return F.avg_pool1d(h.permute(0,2,1),2,2,ceil_mode=False).permute(0,2,1), (length//2).float()

            return h * mask.unsqueeze(2), length.float()
        else:

            # h = self.cran2(h.transpose(1,2))
            # h = h.transpose(1,2)
            h = self.linear2(h)
            h = self.enc2(h)
            h = self.linear3(h)

            if self.training:
                return F.avg_pool1d(h.permute(0,2,1),2,2,ceil_mode=False).permute(0,2,1), (length//2).float()
            
            return h, length.float()

    def _traceback(self, acc_cost_matrix : npt.DTypeLike):
        """ Encontra a path (matching) dado uma matriz de custo acumulado obtida a partir do cálculo do DTW.
        Args:
            acc_cost_matrix (npt.DTypeLike): matriz de custo acumulado obtida a partir do cálculo do DTW.

        Returns:
            Tuple[npt.ArrayLike, npt.ArrayLike]: coordenadas ponto a ponto referente a primeira e segunda sequência utilizada no cálculo do DTW.
        """
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

    def _loss(self, data, lens, n_epoch):
        """ Loss de um batch

        Args:
            data (torch tensor): Batch composto por mini self.nw mini-batches. Cada mini-batch é composto por 16 assinaturas e se refere a um único usuário e é disposto da seguinte forma:
            [0]: âncora; [1:6]: amostras positivas; [6:11]: falsificações profissionais; [11:16]: aleatórias 
            lens (torch tensor (int)): tamanho de cada assinatura no batch
            n_epoch (int): número da época que se está calculando.

        Returns:
            torch.tensor (float): valor da loss
        """
        step = (self.ng + self.nf + 1)
        sep_loss_p = 0
        sep_loss_n = 0
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
            for i in range(len(positives)):
                dist_g[i] = self.sdtw(anchor[None, :int(len_a)], positives[i:i+1, :int(len_p[i])])[0] / (len_a + len_p[i])

            for i in range(len(negatives)):
                dist_n[i] = self.sdtw(anchor[None, :int(len_a)], negatives[i:i+1, :int(len_n[i])])[0] / (len_a + len_n[i])
                
            only_pos = torch.sum(dist_g) * (self.model_lambda /self.ng)
            
            lk = 0
            non_zeros = 1
            for g in dist_g:
                for n in dist_n:
                    temp = F.relu(g + self.margin - n) #+ F.relu(g - 0.7) * 0.5
                    if temp > 0:
                        lk += temp
                        non_zeros+=1

            lk /= non_zeros

            user_loss = lk + only_pos

            total_loss += user_loss

            lp = torch.sum(F.relu(torch.sub(dist_g, self.th)))
            pv = lp / (lp.data.nonzero(as_tuple=False).size(0) + 1)
            sep_loss_p += pv

            ln = torch.sum(F.relu(torch.sub(self.th, dist_n)))
            nv = ln / (ln.data.nonzero(as_tuple=False).size(0) + 1)
            sep_loss_n += nv
        

        sep_loss_p /= self.nw
        sep_loss_n /= self.nw
        total_loss /= self.nw

        return total_loss, sep_loss_p, sep_loss_n
    
    def _triplet_loss(self, dists):
        
        step = (self.ng + self.nf) # ignora a ancora mesmo, isso são as distâncias e não o mini batch
        total_loss = 0

        sep_loss = torch.tensor(0, requires_grad=True)

        for i in range(0, self.nw):

            dist_g = dists[step * i: step * i + self.ng]
            dist_n = dists[step * i + self.ng: step * (i+1)]

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

            lp = torch.sum(F.relu(torch.sub(dist_g, self.th))) 
            pv = lp / (lp.data.nonzero(as_tuple=False).size(0) + 1)
            sep_loss += pv
        
        total_loss /= self.nw
        sep_loss /= self.nw
        return total_loss, sep_loss
        
    def _sep_loss(self, dists):
        step = (self.ng + self.nf) # ignora a ancora mesmo, isso são as distâncias e não o mini batch
        total_loss = 0

        for i in range(0, self.nw):

            dist_g = dists[step * i: step * i + self.ng]
            dist_n = dists[step * i + self.ng: step * (i+1)]
      
            lp = 0

            lp = F.relu(dist_g - self.th) 
            pv = lp / (lp.data.nonzero(as_tuple=False).size(0) + 1)

            # ln = F.relu(self.th - dist_n)
            # nv = ln / (ln.data.nonzero(as_tuple=False).size(0) + 1)
            user_loss = pv 
            # user_loss = pv * 0.5 + nv * 0.5

            total_loss += user_loss
        
        total_loss /= self.nw

        return total_loss

    def get_distances(self, data, lens):
        """ Loss de um batch

        Args:
            data (torch tensor): Batch composto por mini self.nw mini-batches. Cada mini-batch é composto por 16 assinaturas e se refere a um único usuário e é disposto da seguinte forma:
            [0]: âncora; [1:6]: amostras positivas; [6:11]: falsificações profissionais; [11:16]: aleatórias 
            lens (torch tensor (int)): tamanho de cada assinatura no batch
            n_epoch (int): número da época que se está calculando.

        Returns:
            torch.tensor (float): valor da loss
        """
        step = (self.ng + self.nf + 1)

        dists = torch.zeros(self.batch_size - self.nw, dtype=data.dtype, device=data.device)

        for i in range(0, self.nw):
            anchor    = data[i * step]
            positives = data[i * step + 1 : i * step + 1 + self.ng] 
            negatives = data[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf]

            len_a = lens[i * step]
            len_p = lens[i * step + 1 : i * step + 1 + self.ng] 
            len_n = lens[i * step + 1 + self.ng : i * step + 1 + self.ng + self.nf]

            # dist_g = torch.zeros((len(positives)), dtype=data.dtype, device=data.device)
            # dist_n = torch.zeros((len(negatives)), dtype=data.dtype, device=data.device)

            '''Average_Pooling_2,4,6'''
            for j in range(len(positives)):
                # dist_g[i] = self.sdtw(anchor[None, :int(len_a)], positives[i:i+1, :int(len_p[i])])[0] / (len_a + len_p[i])
                dists[j] = self.sdtw(anchor[None, :int(len_a)], positives[j:j+1, :int(len_p[j])])[0] / (len_a + len_p[j])

            for j in range(len(positives), len(positives)+len(negatives)):
                # dist_n[i] = self.sdtw(anchor[None, :int(len_a)], negatives[i:i+1, :int(len_n[i])])[0] / (len_a + len_n[i])
                k = j - len(positives)
                dists[j] = self.sdtw(anchor[None, :int(len_a)], negatives[k:k+1, :int(len_n[k])])[0] / (len_a + len_n[k])

        return dists
    # def _sep_loss(self, data, lens)

    def _dte(self, x, y, len_x, len_y):
        """ DTW entre assinaturas x e y normalizado pelos seus tamanhos * dimensões

        Args:
            x (torch.tensor): Assinatura 1
            y (torch.tensor): Assinatura 2
            len_x (int): Tamanho da assinatura 1
            len_y (int): Tamanho da assinatura 2

        Returns:
            float: DTW normalizado entre as assinaturas
        """
        return self.dtw(x[None, :int(len_x)], y[None, :int(len_y)])[0] /(64* (len_x + len_y))

    def _inference(self, files : str, scenario : str) -> Tuple[float, str, int]:
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

        test_batch, lens = batches_gen.files2array(refs + [sign], scenario=scenario, developtment=CHEAT)

        mask = self.getOutputMask(lens)
        
        mask = Variable(torch.from_numpy(mask)).cuda()
        inputs = Variable(torch.from_numpy(test_batch)).cuda()

        embeddings, lengths = self(inputs.float(), mask, n_epoch)    
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
                        dk += self._dte(refs[i], refs[j], len_refs[i], len_refs[j])
                        count += 1

            dk = dk/(count)
    
        dk_sqrt = math.sqrt(dk)
        
        dists = []
        for i in range(0, len(refs)):
            dists.append(self._dte(refs[i], sign, len_refs[i], len_sign).detach().cpu().numpy()[0])

        dists = np.array(dists) / dk_sqrt

        s_avg = np.mean(dists)
        s_min = min(dists)

        return s_avg + s_min, user_key, result

    def new_evaluate(self, comparison_file : str, n_epoch : int, result_folder : str):
        """ Avaliação da rede conforme o arquivo de comparação

        Args:
            comparison_file (str): path do arquivo que contém as assinaturas a serem comparadas entre si, bem como a resposta da comparação. 0 é positivo (original), 1 é negativo (falsificação).
            n_epoch (int): número que indica após qual época de treinamento a avaliação está sendo realizada.
            result_folder (str): path onde salvar os resultados.
        """

        self.train(mode=False)
        lines = []
        with open(comparison_file, "r") as fr:
            lines = fr.readlines()

        scenario = 'stylus'
        if 'finger' in comparison_file:
            scenario = 'finger'

        if not os.path.exists(result_folder): os.mkdir(result_folder)

        file_name = (comparison_file.split(os.sep)[-1]).split('.')[0]
        print("\n\tAvaliando " + file_name)
        comparison_folder = result_folder + os.sep + file_name
        if not os.path.exists(comparison_folder): os.mkdir(comparison_folder)

        users = {}

        for line in tqdm(lines, "Calculando distâncias..."):
            distance, user_id, true_label = self._inference(line, scenario=scenario, n_epoch=n_epoch)
            
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

            eer, eer_threshold = metrics.get_eer(y_true=users[user]["true_label"], y_scores=users[user]["distances"])
            eers.append(eer)
            local_buffer += user + ", " + "{:.5f}".format(eer) + ", " + "{:.5f}".format(eer_threshold) + "\n"

        print("Obtendo EER global...")
        
        # Calculo do EER global
        eer_global, eer_threshold_global = metrics.get_eer(global_true_label, global_distances, result_folder=comparison_folder, generate_graph=True, n_epoch=n_epoch)

        buffer += "Global, " + "{:.5f}".format(eer_global) + ", " + "{:.5f}".format(eer_threshold_global) + "\n" + local_buffer

        local_eer_mean = np.mean(np.array(eers))
        self.buffer += file_name + ", " + "{:.5f}".format(local_eer_mean) + ", " + "{:.5f}".format(eer_global) + ", " + "{:.5f}".format(eer_threshold_global) + "\n"

        with open(comparison_folder + os.sep + file_name + " epoch=" + str(n_epoch) + ".csv", "w") as fw:
            fw.write(buffer)

        if eer_global < self.best_eer:
            torch.save(self.state_dict(), result_folder + os.sep + "Backup" + os.sep + "best.pt")
        
        self.train(mode=True)

    def start_train(self, n_epochs : int, batch_size : int, comparison_files : List[str], result_folder : str, triplet_loss_w : float = 0.5):
        """ Loop de treinamento

        Args:
            n_epochs (int): Número de épocas que o treinamento deve ocorrer
            batch_size (int): Tamanho do batch de treinamento
            comparison_files (List[str]): Lista com as paths dos arquivos de comparação a serem avaliados durante o treinamento.
            result_folder (str): Path de onde os resultados de avaliação e o backup dos pesos devem ser armazenados.
        """
        
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) 

        losses = [math.inf]*10

        th = -1
        running_loss = 0

        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        if not os.path.exists(result_folder + os.sep + "Backup"):
            os.mkdir(result_folder + os.sep + "Backup")
        bckp_path = result_folder + os.sep + "Backup"

        for i in range(1, n_epochs+1):
            epoch = batches_gen.generate_epoch()
            epoch_size = len(epoch)
            self.loss_value = running_loss/epoch_size
            losses.append(self.loss_value)
            losses = losses[1:]

            if self.loss_value > min(losses) and i > 80:
                print("\n\nEarly stop!")
                break

            pbar = tqdm(total=(epoch_size//(batch_size//16)), position=0, leave=True, desc="Epoch " + str(i) +" PAL: " + "{:.3f}".format(self.loss_value))

            running_loss = 0
            self.mean_eer = 0
            #PAL = Previous Accumulated Loss
            while epoch != []:
                batch, lens, epoch = batches_gen.get_batch_from_epoch(epoch, batch_size)
                
                mask = self.getOutputMask(lens)

                mask = Variable(torch.from_numpy(mask)).cuda()
                inputs = Variable(torch.from_numpy(batch)).cuda()
                
                optimizer.zero_grad()
                outputs, length = self(inputs.float(), mask, i)

                triplet_loss, sep_loss_p, sep_loss_n = self._loss(outputs, length, i)

                loss = None
                if i <= CHANGE_TRAIN_MODE:
                    loss = triplet_loss
                else:
                    loss = (triplet_loss_w * sep_loss_p) + ( (1-triplet_loss_w)*sep_loss_n )
                
                # loss = self._loss(outputs, length, i)
                
                # dists = self.get_distances(outputs, length)
                # triplet_loss = self._triplet_loss(dists)
                # sep_loss = self._sep_loss(dists)
                # loss = (triplet_loss*0.7 + sep_loss*0.3)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
                pbar.update(1)

            pbar.close()

            if (i % 5 == 0 or i > (n_epochs - 3) ):
                for cf in comparison_files:
                    self.new_evaluate(comparison_file=cf, n_epoch=i, result_folder=result_folder)

            if i >= CHANGE_TRAIN_MODE:
                self.cran.requires_grad_(False)
                self.linear.requires_grad_(False)
                self.enc1.requires_grad_(False)

                # _, self.th_loss = metrics.get_eer(self.labels, self.scores)
                # self.labels = []
                # self.scores = []
            
            self.loss_variation.append(running_loss/epoch_size)
            
            lr_scheduler.step()

            torch.save(self.state_dict(), bckp_path + os.sep + "epoch" + str(i) + ".pt")

        # Loss graph
        plt.xlabel("#Epoch")
        plt.ylabel("Loss")
        plt.plot(list(range(0,len(self.loss_variation))), self.loss_variation)
        plt.savefig(result_folder + os.sep + "loss.png")
        plt.cla()
        plt.clf()