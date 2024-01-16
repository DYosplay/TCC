from typing import List, Tuple, Dict, Any
from utils.utilities import define_loss, dump_hyperparameters

import torch.nn.utils as nutils
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
import DTW.dtw_cuda as dtw
import DTW.soft_dtw_cuda as sdtw
import utils.metrics as metrics

import wandb

import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

class DsPipeline(nn.Module):
    def __init__(self, hyperparameters : Dict[str, Any]):    
        super(DsPipeline, self).__init__()

        # Tuneable Hyperparameters
        self.hyperparameters = hyperparameters
        self.margin = torch.nn.Parameter(torch.tensor(hyperparameters['margin']), requires_grad=False)
        self.lr = torch.nn.Parameter(torch.tensor(hyperparameters['learning_rate']), requires_grad=False)
        self.alpha = torch.nn.Parameter(torch.tensor(hyperparameters['alpha']), requires_grad=False)
        self.beta = torch.nn.Parameter(torch.tensor(hyperparameters['beta']), requires_grad=False)
        self.p = torch.nn.Parameter(torch.tensor(hyperparameters['p']), requires_grad=False)
        self.r = torch.nn.Parameter(torch.tensor(hyperparameters['r']), requires_grad=False)
        
        # variáveis de controle
        self.z = hyperparameters['zscore']
        self.use_fdtw = hyperparameters['use_fdtw']
        self.batch_size = hyperparameters['batch_size']
        
        self.n_layers = 2
        self.n_in = 12
        self.n_out = 64
        self.n_hidden = 128
        self.scores = []
        self.labels = []

        # Variáveis que lidam com as métricas/resultados
        self.user_err_avg = 0 
        self.dataset_folder = hyperparameters['dataset_folder']
        self.comparison_data = {}
        self.buffer = "Epoch, mean_local_eer, global_eer, th_global, var_th, amp_th\n"
        self.eer = []
        self.best_eer = math.inf
        self.last_eer = math.inf
        self.loss_variation = []

        # Definição da rede
        self.cran  = nn.Sequential(
        nn.Conv1d(in_channels=self.n_in, out_channels=self.n_out, kernel_size=7, stride=1, padding=3, bias=True),
        nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True), 
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channels=self.n_out, out_channels=self.n_hidden, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1)
        )

        self.rnn = nn.GRU(self.n_hidden, self.n_hidden, self.n_layers, dropout=0.1, batch_first=True, bidirectional=False)

        self.h0 = Variable(torch.zeros(self.n_layers, self.batch_size, self.n_hidden).cuda(), requires_grad=False)
        self.h1 = Variable(torch.zeros(self.n_layers, 5, self.n_hidden).cuda(), requires_grad=False)
        self.h2 = Variable(torch.zeros(self.n_layers, 2, self.n_hidden).cuda(), requires_grad=False)

        # Fecha a update gate (pra virar uma GARU)
        for i in range(self.n_layers):
            eval("self.rnn.bias_hh_l%d"%i)[self.n_hidden:2*self.n_hidden].data.fill_(-1e10) #Initial update gate bias
            eval("self.rnn.bias_ih_l%d"%i)[self.n_hidden:2*self.n_hidden].data.fill_(-1e10) #Initial update gate bias
    
        self.linear = nn.Linear(self.n_hidden, 64, bias=False)

        nn.init.kaiming_normal_(self.linear.weight, a=1) 
        nn.init.kaiming_normal_(self.cran[0].weight, a=0)
        nn.init.kaiming_normal_(self.cran[3].weight, a=0)
        nn.init.zeros_(self.cran[0].bias)
        nn.init.zeros_(self.cran[3].bias)

        self.loss_function = define_loss(loss_type=hyperparameters['loss_type'], ng=hyperparameters['ng'], nf=hyperparameters['nf'], nw=hyperparameters['nw'], margin=self.margin, model_lambda=hyperparameters['model_lambda'], alpha=self.alpha, beta=self.beta, p=self.p, r=self.r, mmd_kernel_num=hyperparameters['mmd_kernel_num'], mmd_kernel_mul=hyperparameters['mmd_kernel_mul'], margin_max=hyperparameters['margin_max'], margin_min=hyperparameters['margin_min'])
        
        self.dtw = dtw.DTW(True, normalize=False, bandwidth=1)

        # Wandb
        run = None
        if self.hyperparameters['wandb_name'] is not None:
            # wandb.login()

            try:
                if self.hyperparameters['wandb_project_name'] != "":
                    run = wandb.init(
                        project=self.hyperparameters["wandb_project_name"],
                        name=self.hyperparameters["wandb_name"],
                        config=self.hyperparameters,
                    )
                else:
                    run = wandb.init(
                        project=self.hyperparameters["wandb_name"],
                        config=self.hyperparameters,
                    )

                wandb.watch(self, log_freq=100)
            except:
                self.hyperparameters['wandb_name'] = None
    
    def __del__(self):
        if self.hyperparameters['wandb_name'] is not None: wandb.finish()

    def getOutputMask(self, lens):    
        lens = np.array(lens, dtype=np.int32)
        lens = (lens + 1) // 2
        N = len(lens); D = np.max(lens)
        mask = np.zeros((N, D), dtype=np.float32)
        for i in range(N):
            mask[i, 0:lens[i]] = 1.0
        return mask
    
    def forward(self, x, mask, n_epoch):
        length = torch.sum(mask, dim=1)
        length, indices = torch.sort(length, descending=True)
        x = torch.index_select(x, 0, indices)
        mask = torch.index_select(mask, 0, indices)

        h = self.cran(x)
        h = h.transpose(1,2)
        h = h * mask.unsqueeze(2)

        h = nutils.rnn.pack_padded_sequence(h, list(length.cpu().numpy()), batch_first=True)
        if len(x) == self.batch_size: h, _ = self.rnn(h, self.h0)
        elif len(x) > 2: h, _ = self.rnn(h, self.h1)
        else: h, _ = self.rnn(h, self.h2)
        
        h, length = nutils.rnn.pad_packed_sequence(h, batch_first=True) 
        length = Variable(length).cuda()

        '''Recover the original order'''
        _, indices = torch.sort(indices, descending=False)
        h = torch.index_select(h, 0, indices)
        length = torch.index_select(length, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        
        h = self.linear(h)

        if self.training:
            return F.avg_pool1d(h.permute(0,2,1),2,2,ceil_mode=False).permute(0,2,1), (length//2).float()

        return h * mask.unsqueeze(2), length.float()

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
        return self.dtw(x[None, :int(len_x)], y[None, :int(len_y)])[0] /((len_x + len_y))


        # return self.sdtw2(x[None, :int(len_x)], y[None, :int(len_y)])[0] /((len_x + len_y)) - ((self.sdtw2(x[None, :int(len_x)], x[None, :int(len_x)])[0] /((len_x + len_x))) + (self.sdtw2(y[None, :int(len_y)], y[None, :int(len_y)])[0] /((len_y + len_y))))/2

        # if self.use_fdtw:
        #     distance, _ = fastdtw(x[:int(len_x)].detach().cpu().numpy(), y[:int(len_y)].detach().cpu().numpy(), dist=2)
        #     return torch.tensor([distance /((len_x + len_y))])
        # else:
        #     return self.dtw(x[None, :int(len_x)], y[None, :int(len_y)])[0] /((len_x + len_y))
            # return self.dtw(x[None, :int(len_x)], y[None, :int(len_y)])[0] /(64*(len_x + len_y))
            # return self.dtw((x[None, :int(len_x)]-torch.mean(x))/(torch.max(x)-torch.min(x)), (y[None, :int(len_y)]-torch.mean(y))/(torch.max(y)-torch.min(y)))[0] /(64*(len_x + len_y))

    def _inference(self, files : str, n_epoch : int) -> Tuple[float, str, int]:
        """
        Args:
            files (str): string no formato: ref1 [,ref2, ref3, ref4], sign, label 

        Raises:
            ValueError: "Arquivos de comparação com formato desconhecido"

        Returns:
            float, str, int: distância da assinatura, usuário, label 
        """
        scenario = "stylus"
        if "finger" in files:
            scenario = "finger"

        tokens = files.split(" ")
        user_key = tokens[0].split("_")[0]
        
        result = math.nan
        refs = []
        sign = ""
        s_avg = 0
        s_min = 0

        if len(tokens) == 2:
            a = tokens[0].split('_')[0]
            b = tokens[1].split('_')[0]
            result = 0 if a == b and '_g_' in tokens[1] else 1; refs.append(tokens[0]); sign = tokens[1]
        elif len(tokens) == 3: result = int(tokens[2]); refs.append(tokens[0]); sign = tokens[1]
        elif len(tokens) == 6: result = int(tokens[5]); refs = tokens[0:4]; sign = tokens[4]
        else: raise ValueError("Arquivos de comparação com formato desconhecido")

        # if 'u0148' in refs[0] or 'u0272' in refs[0]:

        test_batch, lens = batches_gen.files2array(refs + [sign], z=self.z, developtment=False, scenario=scenario)

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
                        dk += (self._dte(refs[i], refs[j], len_refs[i], len_refs[j]))
                        count += 1

            dk = dk/(count)
    
        dk_sqrt = math.sqrt(dk)
        
        dists = []
        for i in range(0, len(refs)):
            dists.append(self._dte(refs[i], sign, len_refs[i], len_sign).detach().cpu().numpy()[0])

        dists = np.array(dists) / dk_sqrt
        s_avg = np.mean(dists)
        s_min = min(dists)

        return (s_avg + s_min), user_key, result

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

        if not os.path.exists(result_folder): os.mkdir(result_folder)

        file_name = (comparison_file.split(os.sep)[-1]).split('.')[0]
        print("\n\tAvaliando " + file_name)
        comparison_folder = result_folder + os.sep + file_name
        if not os.path.exists(comparison_folder): os.mkdir(comparison_folder)

        users = {}

        for line in tqdm(lines, "Calculando distâncias..."):
            distance, user_id, true_label = self._inference(line, n_epoch=n_epoch)
            
            if user_id not in users: 
                users[user_id] = {"distances": [distance], "true_label": [true_label], "predicted_label": []}
            else:
                users[user_id]["distances"].append(distance)
                users[user_id]["true_label"].append(true_label)

        # Nesse ponto, todos as comparações foram feitas
        buffer = "user, eer_local, threshold, mean_eer, var_th, amp_th\n"
        local_buffer = ""
        global_true_label = []
        global_distances = []

        eers = []

        local_ths = []

        # Calculo do EER local por usuário:
        for user in tqdm(users, desc="Obtendo EER local..."):
            global_true_label += users[user]["true_label"]
            global_distances  += users[user]["distances"]

            # if "Task" not in comparison_file:
            if 0 in users[user]["true_label"] and 1 in users[user]["true_label"]:
                eer, eer_threshold = metrics.get_eer(y_true=users[user]["true_label"], y_scores=users[user]["distances"])

                local_ths.append(eer_threshold)
                eers.append(eer)
                local_buffer += user + ", " + "{:.5f}".format(eer) + ", " + "{:.5f}".format(eer_threshold) + ", 0, 0, 0\n"

        print("Obtendo EER global...")
        
        # Calculo do EER global
        eer_global, eer_threshold_global = metrics.get_eer(global_true_label, global_distances, result_folder=comparison_folder, generate_graph=True, n_epoch=n_epoch)

        local_eer_mean = np.mean(np.array(eers))
        local_ths = np.array(local_ths)
        local_ths_var  = np.var(local_ths)
        local_ths_amp  = np.max(local_ths) - np.min(local_ths)

        buffer += "Global, " + "{:.5f}".format(eer_global) + ", " + "{:.5f}".format(eer_threshold_global) + ", " + "{:.5f}".format(local_eer_mean) + ", " + "{:.5f}".format(local_ths_var) + ", " + "{:.5f}".format(local_ths_amp) + "\n" + local_buffer

        self.buffer += str(n_epoch) + ", " + "{:.5f}".format(local_eer_mean) + ", " + "{:.5f}".format(eer_global) + ", " + "{:.5f}".format(eer_threshold_global) + ", " + "{:.5f}".format(local_ths_var) + ", " + "{:.5f}".format(local_ths_amp) + "\n"

        with open(comparison_folder + os.sep + file_name + " epoch=" + str(n_epoch) + ".csv", "w") as fw:
            fw.write(buffer)


        self.last_eer = eer_global

        if n_epoch != 0 and n_epoch != 777 and n_epoch != 888:
            if eer_global < self.best_eer:
                torch.save(self.state_dict(), result_folder + os.sep + "Backup" + os.sep + "best.pt")
                self.best_eer = eer_global
                print("EER atualizado: " + str(self.best_eer))

        self.train(mode=True)

        ret_metrics = {"Global EER": eer_global, "Mean Local EER": local_eer_mean, "Global Threshold": eer_threshold_global, "Local Threshold Variance": local_ths_var, "Local Threshold Amplitude": local_ths_amp}
        if self.hyperparameters['wandb_name'] is not None: wandb.log(ret_metrics)
        
        return ret_metrics

    def start_train(self, comparison_files : List[str], result_folder : str):
        """ Loop de treinamento

        Args:
            comparison_files (List[str]): Lista com as paths dos arquivos de comparação a serem avaliados durante o treinamento.
            result_folder (str): Path de onde os resultados de avaliação e o backup dos pesos devem ser armazenados.
        """
        dump_hyperparameters(hyperparameters=self.hyperparameters, res_folder=result_folder)

        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.hyperparameters['momentum'])
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hyperparameters['decay']) 
            
        if not os.path.exists(result_folder): os.mkdir(result_folder)

        if not os.path.exists(result_folder + os.sep + "Backup"): os.mkdir(result_folder + os.sep + "Backup")
        bckp_path = result_folder + os.sep + "Backup"

        running_loss = 0
        for i in range(1, self.hyperparameters['epochs']+1):
            epoch = None

            if self.hyperparameters['dataset_scenario'] == "mix":
                epoch = batches_gen.generate_mixed_epoch()
            elif self.hyperparameters['dataset_scenario'] == "stylus":
                epoch = batches_gen.generate_epoch()
            elif self.hyperparameters['dataset_scenario'] == "finger":
                epoch = batches_gen.generate_epoch(dataset_folder="../Data/DeepSignDB/Development/finger", train_offset=[(1009, 1084)], scenario='finger')

            epoch_size = len(epoch)
            self.loss_value = running_loss/epoch_size

            if (self.best_eer > 0.04 and i >= self.hyperparameters['early_stop']):
                print("\n\nEarly stop!")
                break

            pbar = tqdm(total=(epoch_size//(self.batch_size//16)), position=0, leave=True, desc="Epoch " + str(i) +" PAL: " + "{:.4f}".format(self.loss_value))

            running_loss = 0
            #PAL = Previous Accumulated Loss
            while epoch != []:
            # if True:
                batch, lens, epoch = batches_gen.get_batch_from_epoch(epoch, self.batch_size, z=self.z)
                
                mask = self.getOutputMask(lens)
                mask = Variable(torch.from_numpy(mask)).cuda()
                inputs = Variable(torch.from_numpy(batch)).cuda()
                
                outputs, length = self(inputs.float(), mask, i)
                
                optimizer.zero_grad()

                loss = self.loss_function(outputs, length)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()
            
                pbar.update(1)

            pbar.close()

            if self.hyperparameters['wandb_name'] is not None: wandb.log({'loss': running_loss/epoch_size}) 

            if (i % self.hyperparameters['eval_step'] == 0 or i > (self.hyperparameters['epochs'] - 3) ):
                for cf in comparison_files:
                    self.new_evaluate(comparison_file=cf, n_epoch=i, result_folder=result_folder)
                    
            
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

    def start_transfer(self, comparison_files : List[str], result_folder : str, teacher_model):
        """ Loop de treinamento

        Args:
            comparison_files (List[str]): Lista com as paths dos arquivos de comparação a serem avaliados durante o treinamento.
            result_folder (str): Path de onde os resultados de avaliação e o backup dos pesos devem ser armazenados.
        """
        dump_hyperparameters(hyperparameters=self.hyperparameters, res_folder=result_folder)

        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.hyperparameters['momentum'])
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hyperparameters['decay']) 
            
        if not os.path.exists(result_folder): os.mkdir(result_folder)

        if not os.path.exists(result_folder + os.sep + "Backup"): os.mkdir(result_folder + os.sep + "Backup")
        bckp_path = result_folder + os.sep + "Backup"

        running_loss = 0
        for i in range(1, self.hyperparameters['epochs']+1):
            epoch = None

            if self.hyperparameters['dataset_scenario'] == "mix":
                epoch = batches_gen.generate_mixed_epoch()
            elif self.hyperparameters['dataset_scenario'] == "stylus":
                epoch = batches_gen.generate_epoch()
            elif self.hyperparameters['dataset_scenario'] == "finger":
                epoch = batches_gen.generate_epoch(dataset_folder="../Data/DeepSignDB/Development/finger", train_offset=[(1009, 1084)], scenario='finger')

            epoch_size = len(epoch)
            self.loss_value = running_loss/epoch_size

            if (self.best_eer > 0.04 and i >= self.hyperparameters['early_stop']):
                print("\n\nEarly stop!")
                break

            pbar = tqdm(total=(epoch_size//(self.batch_size//16)), position=0, leave=True, desc="Epoch " + str(i) +" PAL: " + "{:.4f}".format(self.loss_value))

            running_loss = 0
            #PAL = Previous Accumulated Loss
            while epoch != []:
            # if True:
                batch, lens, epoch = batches_gen.get_batch_from_epoch(epoch, self.batch_size, z=self.z)
                
                mask = self.getOutputMask(lens)
                mask = Variable(torch.from_numpy(mask)).cuda()
                inputs = Variable(torch.from_numpy(batch)).cuda()
                
                outputs, length = self(inputs.float(), mask, i)
                outputs_t, length_t = teacher_model(inputs.float(), mask, i)
                
                optimizer.zero_grad()

                loss = self.loss_function(outputs, length, outputs_t)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()
            
                pbar.update(1)

            pbar.close()

            if self.hyperparameters['wandb_name'] is not None: wandb.log({'loss': running_loss/epoch_size}) 

            if (i % self.hyperparameters['eval_step'] == 0 or i > (self.hyperparameters['epochs'] - 3) ):
                for cf in comparison_files:
                    self.new_evaluate(comparison_file=cf, n_epoch=i, result_folder=result_folder)
                    
            
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