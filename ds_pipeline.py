from typing import List, Tuple, Dict, Any
from utils.utilities import define_loss, dump_hyperparameters
from utils.pre_alignment import align
import gc
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

import pickle

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
        if hyperparameters['ga']: self.p.requires_grad_()
        self.r = torch.nn.Parameter(torch.tensor(hyperparameters['r']), requires_grad=False)
        # self.q = torch.nn.Parameter(torch.tensor(hyperparameters['q']), requires_grad=False)
        self.q =hyperparameters['q']
        
        # variáveis de controle
        self.z = hyperparameters['zscore']
        self.use_fdtw = hyperparameters['use_fdtw']
        self.batch_size = hyperparameters['batch_size']
        
        self.n_layers = 2
        self.n_in = 12
        self.n_out = hyperparameters["nout"]
        self.n_hidden = hyperparameters["nhidden"]
        self.scores = []
        self.labels = []

        self.dumped_users = {}
        self.knn_matrix = np.zeros((443,443)) - 1

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

        self.loss_function = define_loss(loss_type=hyperparameters['loss_type'], ng=hyperparameters['ng'], nf=hyperparameters['nf'], nw=hyperparameters['nw'], margin=self.margin, model_lambda=hyperparameters['model_lambda'], alpha=self.alpha, beta=self.beta, p=self.p, r=self.r, q=self.q, mmd_kernel_num=hyperparameters['mmd_kernel_num'], mmd_kernel_mul=hyperparameters['mmd_kernel_mul'], margin_max=hyperparameters['margin_max'], margin_min=hyperparameters['margin_min'], nsl=hyperparameters['number_of_slices'])
        
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
        if self.hyperparameters['number_of_slices'] > 0:
            d = 0
            x_step = int(len_x//self.hyperparameters['number_of_slices'])
            y_step = int(len_y//self.hyperparameters['number_of_slices'])
            for n in range(1, int(self.hyperparameters['number_of_slices'])):
                d += self.dtw(x[None, x_step*(n-1):x_step*n], y[None, y_step*(n-1):y_step*n])[0] /(64*(x_step + y_step))


            n = int(self.hyperparameters['number_of_slices'])
            d5 = self.dtw(x[None, x_step*(n-1):int(len_x)], y[None, y_step*(n-1):int(len_y)])[0] /(64* ((len_x - x_step*(n-1)) + (len_y - y_step*(n-1))))

            d += d5
            # f = self.dtw(x[None, :int(len_x)], y[None, :int(len_y)])[0] /(64*(len_x + len_y))
            return d
        
        if self.hyperparameters['xnorm']:
            return self.dtw(x[None, :int(len_x)], y[None, :int(len_y)])[0] /(64*(len_x))
         
        return self.dtw(x[None, :int(len_x)], y[None, :int(len_y)])[0] /(64*(len_x + len_y))

        # return self.sdtw2(x[None, :int(len_x)], y[None, :int(len_y)])[0] /((len_x + len_y)) - ((self.sdtw2(x[None, :int(len_x)], x[None, :int(len_x)])[0] /((len_x + len_x))) + (self.sdtw2(y[None, :int(len_y)], y[None, :int(len_y)])[0] /((len_y + len_y))))/2

        # if self.use_fdtw:
        #     distance, _ = fastdtw(x[:int(len_x)].detach().cpu().numpy(), y[:int(len_y)].detach().cpu().numpy(), dist=2)
        #     return torch.tensor([distance /((len_x + len_y))])
        # else:
        #     return self.dtw(x[None, :int(len_x)], y[None, :int(len_y)])[0] /((len_x + len_y))
            # return self.dtw(x[None, :int(len_x)], y[None, :int(len_y)])[0] /(64*(len_x + len_y))
            # return self.dtw((x[None, :int(len_x)]-torch.mean(x))/(torch.max(x)-torch.min(x)), (y[None, :int(len_y)]-torch.mean(y))/(torch.max(y)-torch.min(y)))[0] /(64*(len_x + len_y))

    def _inference(self, files : str, n_epoch : int, result_folder : str = None) -> Tuple[float, str, int]:
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
        files_bckp = tokens.copy()
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
        users = refs.copy()

        mask = self.getOutputMask(lens)
        
        mask = Variable(torch.from_numpy(mask)).cuda()
        inputs = Variable(torch.from_numpy(test_batch)).cuda()

        embeddings, lengths = self(inputs.float(), mask, n_epoch)    
        refs = embeddings[:len(embeddings)-1]
        sign = embeddings[-1]

        len_refs = lengths[:len(embeddings)-1]
        len_sign = lengths[-1]

        if self.hyperparameters['generate_features']:
            if not os.path.exists(result_folder + os.sep + "features"):
                os.makedirs(result_folder + os.sep + "features")
            if user_key not in self.dumped_users:
                self.dumped_users[user_key] = True

                for i in range(len(refs)):
                    file_name = files_bckp[i].split('.')[0] + ".pt"
                    torch.save(refs[i][:int(len_refs[i])], result_folder + os.sep + "features" + os.sep + file_name)

            file_name = files_bckp[-2].split('.')[0] + ".pt"
            torch.save(sign[:int(len_sign)], result_folder + os.sep + "features" + os.sep + file_name)

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
    
        if dk <= 0 or dk is math.nan or dk > 10000:
            print("oi")
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
            distance, user_id, true_label = self._inference(line, n_epoch=n_epoch, result_folder=result_folder)
            
            if user_id not in users: 
                users[user_id] = {"distances": [distance], "true_label": [true_label], "predicted_label": []}
            else:
                users[user_id]["distances"].append(distance)
                users[user_id]["true_label"].append(true_label)

        # Nesse ponto, todos as comparações foram feitas
        buffer = "user, eer_local, threshold, mean_eer, var_th, amp_th, th_range\n"
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
                th_range_local = np.max(np.array(users[user]["distances"])[np.array(users[user]["distances"]) < eer_threshold])

                local_ths.append(eer_threshold)
                eers.append(eer)
                local_buffer += user + ", " + "{:.5f}".format(eer) + ", " + "{:.5f}".format(eer_threshold) + ", 0, 0, 0, " + "{:.5f}".format(eer_threshold -th_range_local) + " (" + "{:.5f}".format(th_range_local) + "~" + "{:.5f}".format(eer_threshold) + ")\n"

        print("Obtendo EER global...")
        
        # Calculo do EER global
        eer_global, eer_threshold_global = metrics.get_eer(global_true_label, global_distances, result_folder=comparison_folder, generate_graph=True, n_epoch=n_epoch)

        local_eer_mean = np.mean(np.array(eers))
        local_ths = np.array(local_ths)
        local_ths_var  = np.var(local_ths)
        local_ths_amp  = np.max(local_ths) - np.min(local_ths)
        
        th_range_global = np.max(np.array(global_distances)[np.array(global_distances) < eer_threshold_global])

        buffer += "Global, " + "{:.5f}".format(eer_global) + ", " + "{:.5f}".format(eer_threshold_global) + ", " + "{:.5f}".format(local_eer_mean) + ", " + "{:.5f}".format(local_ths_var) + ", " + "{:.5f}".format(local_ths_amp) + ", " + "{:.5f}".format(eer_threshold_global -th_range_global) + " (" + "{:.5f}".format(th_range_global) + "~" + "{:.5f}".format(eer_threshold_global) + ")\n" + local_buffer

        self.buffer += str(n_epoch) + ", " + "{:.5f}".format(local_eer_mean) + ", " + "{:.5f}".format(eer_global) + ", " + "{:.5f}".format(eer_threshold_global) + ", " + "{:.5f}".format(local_ths_var) + ", " + "{:.5f}".format(local_ths_amp) + ", " + "{:.5f}".format(eer_threshold_global -th_range_global) + " (" + "{:.5f}".format(th_range_global) + "~" + "{:.5f}".format(eer_threshold_global) + ")\n"

        with open(comparison_folder + os.sep + file_name + " epoch=" + str(n_epoch) + ".csv", "w") as fw:
            fw.write(buffer)


        self.last_eer = eer_global
        ret_metrics = {"Global EER": eer_global, "Mean Local EER": local_eer_mean, "Global Threshold": eer_threshold_global, "Local Threshold Variance": local_ths_var, "Local Threshold Amplitude": local_ths_amp}
        if n_epoch != 0 and n_epoch != 777 and n_epoch != 888 and n_epoch < 100:
            if eer_global < self.best_eer:
                torch.save(self.state_dict(), result_folder + os.sep + "Backup" + os.sep + "best.pt")
                self.best_eer = eer_global
                print("EER atualizado: ")
                print(ret_metrics)

        self.train(mode=True)

        # ret_metrics = {"Global EER": eer_global, "Mean Local EER": local_eer_mean, "Global Threshold": eer_threshold_global, "Local Threshold Variance": local_ths_var, "Local Threshold Amplitude": local_ths_amp}
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
                epoch = batches_gen.generate_epoch(dataset_folder= self.hyperparameters['dataset_folder'] + os.sep + "Development" + os.sep + self.hyperparameters['dataset_scenario'], hyperparameters = self.hyperparameters)
            elif self.hyperparameters['dataset_scenario'] == "finger":
                epoch = batches_gen.generate_epoch(dataset_folder= self.hyperparameters['dataset_folder'] + os.sep + "Development" + os.sep + self.hyperparameters['dataset_scenario'], train_offset=[(1009, 1084)], scenario='finger', hyperparameters = self.hyperparameters)

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
                batch, lens, epoch = batches_gen.get_batch_from_epoch(epoch, self.batch_size, z=self.z, hyperparameters=self.hyperparameters)
                
                mask = self.getOutputMask(lens)
                mask = Variable(torch.from_numpy(mask)).cuda()
                inputs = Variable(torch.from_numpy(batch)).cuda()
                
                outputs, length = self(inputs.float(), mask, i)
                
                optimizer.zero_grad()

                loss = self.loss_function(outputs, length)
                loss.backward()

                optimizer.step()
                if self.hyperparameters['ga']: self.p.data.clamp_(0.0,1.0)

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

                if self.hyperparameters['ga']: self.p.data.clamp_(0.0,1.0)
                print(self.p)

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

    def knn_generate_matrix(self, result_folder : str, n_epoch : int = 0):
        folder = self.hyperparameters["dataset_folder"] + os.sep + "Evaluation" + os.sep + self.hyperparameters['dataset_scenario']
        files = sorted(os.listdir(folder))

        dists = {}

        assert len(files) >= self.hyperparameters['q']
        assert self.hyperparameters['p'] >= 0

        print("Calculando distâncias...")
        # limit = len(files)
        limit = 5001
        for i in tqdm(range(int(self.hyperparameters['p']), int(self.hyperparameters['q']))): # index start for paralelization
        # for i in tqdm(range(0, len(files))):
            
            for j in tqdm(range(i, limit)):
            
            # if not ('0319' in files[i]):
            #     continue
            # for j in tqdm(range(0, len(files))):
            
                batch = [files[i], files[j]]

                # prepara chaves do dicionário
                ref_id = files[i]
                tst_id = files[j]

                if '_s_' in ref_id and '_s_' in tst_id:
                    continue

                test_batch, lens = batches_gen.files2array(batch, z=self.z, developtment=False, scenario=self.hyperparameters['dataset_scenario'])
            
                mask = self.getOutputMask(lens)
        
                mask = Variable(torch.from_numpy(mask)).cuda()
                inputs = Variable(torch.from_numpy(test_batch)).cuda()

                embeddings, lengths = self(inputs.float(), mask, n_epoch)    
                refs = embeddings[:len(embeddings)-1]
                sign = embeddings[-1]

                len_refs = lengths[:len(embeddings)-1]
                len_sign = lengths[-1]

                distance_value = (self._dte(refs[0], sign, len_refs[0], len_sign).detach().cpu().numpy()[0])

                if ref_id not in dists:
                    dists[ref_id] = {batch[1]: distance_value}
                else:
                    dists[ref_id][batch[1]] = distance_value

                if ref_id != tst_id:
                    if tst_id not in dists:
                        dists[tst_id] = {batch[0]: distance_value}
                    else:
                        dists[tst_id][batch[0]] = distance_value

        print("Ordenando distâncias")
        for key in tqdm(dists.keys()):
            dists[key] = dict(sorted(dists[key].items(), key=lambda item: item[1]))

        # Open the file in binary mode
        with open(result_folder + os.sep + "matrix_" + str(int(self.hyperparameters['p'])) + "_to_" + str(int(self.hyperparameters['q'])) + ".pickle", 'wb') as fw:
            # Serialize and write the variable to the file
            pickle.dump(dists, fw)
    
    def knn(self, matrix_path : str, comparison_file : str, result_folder : str, n_epoch : int = 0):
        dists : Dict[Dict[str, float]]
        with open(matrix_path, 'rb') as fr:
            # Serialize and write the variable to the file
            dists = pickle.load(fr) 
        
        # merge nos dicts
        # ordenar dicts

        lines = []
        with open(comparison_file, "r") as fr:
            lines = fr.readlines()

        if not os.path.exists(result_folder): os.mkdir(result_folder)

        file_name = (comparison_file.split(os.sep)[-1]).split('.')[0]
        print("\n\tAvaliando " + file_name)
        comparison_folder = result_folder + os.sep + file_name
        if not os.path.exists(comparison_folder): os.mkdir(comparison_folder)

        users = {}

        for line in lines:
            files = line.split(' ')

            assert len(files) == 6

            refs = files[:4]
            sign = files[-2]
            true_label  = files[-1]

            dt = np.mean(np.array(list(dists[sign].values())[1:4]))

            dt = 0
            dtd = 0

            u = refs[0].split("_")[0] + "_g_"
            l = list(dists[sign].keys())
            for i in range(len(l)):
                # ignora se eh original, mas nao eh referencia
                if u in l[i] and ('v00' not in l[i]) and ('v01' not in l[i]) and ('v02' not in l[i]) and ('v03' not in l[i]):
                    continue

                dt += dists[sign][l[i]]
                dtd += 1
                if dtd == 3: break
            dt /= dtd



            # dr = (dists[user_id][refs[0]] + dists[user_id][refs[1]] + dists[user_id][refs[2]] + dists[user_id][refs[3]]) / 3
            dr = 0
            drd = 0
            for i in range(0, 4):
                for j in range(i+1, 4):
                    dr += dists[refs[i]][refs[j]]
                    drd += 1
            dr = dr/drd
           
            distance = abs(dr-dt)

            user_id = refs[0].split('_')[0] + 'g'
            
            user_tst = sign.split('_')[0]
            if '_g_' in sign: user_tst += 'g'
            else: user_tst += 's'

            if user_id not in users: 
                users[user_id] = {"distances": [distance], "true_label": [true_label], "predicted_label": []}
            else:
                users[user_id]["distances"].append(distance)
                users[user_id]["true_label"].append(true_label)

        # Nesse ponto, todos as comparações foram feitas
        buffer = "user, eer_local, threshold, mean_eer, var_th, amp_th, th_range\n"
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
                th_range_local = np.max(np.array(users[user]["distances"])[np.array(users[user]["distances"]) < eer_threshold])

                local_ths.append(eer_threshold)
                eers.append(eer)
                local_buffer += user + ", " + "{:.5f}".format(eer) + ", " + "{:.5f}".format(eer_threshold) + ", 0, 0, 0, " + "{:.5f}".format(eer_threshold -th_range_local) + " (" + "{:.5f}".format(th_range_local) + "~" + "{:.5f}".format(eer_threshold) + ")\n"

        print("Obtendo EER global...")
        
        # Calculo do EER global
        eer_global, eer_threshold_global = metrics.get_eer(global_true_label, global_distances, result_folder=comparison_folder, generate_graph=True, n_epoch=n_epoch)

        local_eer_mean = np.mean(np.array(eers))
        local_ths = np.array(local_ths)
        local_ths_var  = np.var(local_ths)
        local_ths_amp  = np.max(local_ths) - np.min(local_ths)
        
        th_range_global = np.max(np.array(global_distances)[np.array(global_distances) < eer_threshold_global])

        buffer += "Global, " + "{:.5f}".format(eer_global) + ", " + "{:.5f}".format(eer_threshold_global) + ", " + "{:.5f}".format(local_eer_mean) + ", " + "{:.5f}".format(local_ths_var) + ", " + "{:.5f}".format(local_ths_amp) + ", " + "{:.5f}".format(eer_threshold_global -th_range_global) + " (" + "{:.5f}".format(th_range_global) + "~" + "{:.5f}".format(eer_threshold_global) + ")\n" + local_buffer

        self.buffer += str(n_epoch) + ", " + "{:.5f}".format(local_eer_mean) + ", " + "{:.5f}".format(eer_global) + ", " + "{:.5f}".format(eer_threshold_global) + ", " + "{:.5f}".format(local_ths_var) + ", " + "{:.5f}".format(local_ths_amp) + ", " + "{:.5f}".format(eer_threshold_global -th_range_global) + " (" + "{:.5f}".format(th_range_global) + "~" + "{:.5f}".format(eer_threshold_global) + ")\n"

        with open(comparison_folder + os.sep + file_name + " epoch=" + str(n_epoch) + ".csv", "w") as fw:
            fw.write(buffer)


        self.last_eer = eer_global
        ret_metrics = {"Global EER": eer_global, "Mean Local EER": local_eer_mean, "Global Threshold": eer_threshold_global, "Local Threshold Variance": local_ths_var, "Local Threshold Amplitude": local_ths_amp}
        if n_epoch != 0 and n_epoch != 777 and n_epoch != 888 and n_epoch < 100:
            if eer_global < self.best_eer:
                torch.save(self.state_dict(), result_folder + os.sep + "Backup" + os.sep + "best.pt")
                self.best_eer = eer_global
                print("EER atualizado: ")
                print(ret_metrics)

        # ret_metrics = {"Global EER": eer_global, "Mean Local EER": local_eer_mean, "Global Threshold": eer_threshold_global, "Local Threshold Variance": local_ths_var, "Local Threshold Amplitude": local_ths_amp}
        if self.hyperparameters['wandb_name'] is not None: wandb.log(ret_metrics)
        
        return ret_metrics