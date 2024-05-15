import torch
import torch.nn as nn

from tqdm import tqdm
import os
import random

import DTW.soft_dtw_cuda as soft_dtw
import DTW.dtw_cuda as dtw

import numpy as np
from matplotlib import pyplot as plt

import utils.parse_arguments as parse_arguments
import argparse
import math
from sklearn.metrics import roc_curve

from utils.constants import *
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

device = torch.device("cuda")
# dataset_folder = os.path.join("..","Resultados", "ROT_X2_", "ROT_X2_005", "generated_features")
dataset_folder = os.path.join("ROT_X2_", "ROT_X2_005", "generated_features")
training_guide = "training_guide.txt"

def get_eer(y_true, y_scores, result_folder : str = None, generate_graph : bool = False, n_epoch : int = None):
    fpr, tpr, threshold = roc_curve(y_true=y_true, y_score=y_scores, pos_label=1)
    fnr = 1 - tpr

    far = fpr
    frr = fnr

    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # as a sanity check the value should be close to
    eer2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    eer = (eer + eer2)/2
    # eer = min(eer, eer2)

    if generate_graph:
        frr_list = np.array(frr)
        far_list = np.array(far)

        plt.plot(threshold, frr_list, 'b', label="FRR")
        plt.plot(threshold, far_list, 'r', label="FAR")
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

class conbr_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation = dilation, padding=3, bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)
        
        return out       

class se_block(nn.Module):
    def __init__(self,in_layer, out_layer):
        super(se_block, self).__init__()
        
        self.conv1 = nn.Conv1d(in_layer, out_layer//8, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_layer//8, in_layer, kernel_size=1, padding=0)
        self.fc = nn.Linear(1,out_layer//8)
        self.fc2 = nn.Linear(out_layer//8,out_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):

        x_se = nn.functional.adaptive_avg_pool1d(x,1)
        x_se = self.conv1(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv2(x_se)
        x_se = self.sigmoid(x_se)
        
        x_out = torch.add(x, x_se)
        return x_out

class re_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, dilation):
        super(re_block, self).__init__()
        
        self.cbr1 = conbr_block(in_layer,out_layer, kernel_size, 1, dilation)
        self.cbr2 = conbr_block(out_layer,out_layer, kernel_size, 1, dilation)
        self.seblock = se_block(out_layer, out_layer)
    
    def forward(self,x):

        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.seblock(x_re)
        x_out = torch.add(x, x_re)
        return x_out          

class UNET_1D(nn.Module):
    def __init__(self ,input_dim,layer_n,kernel_size,depth):
        super(UNET_1D, self).__init__()
        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth
        
        self.AvgPool1D1 = nn.AvgPool1d(1, stride=2,ceil_mode=False)
        self.AvgPool1D2 = nn.AvgPool1d(1, stride=4,ceil_mode=False)
        self.AvgPool1D3 = nn.AvgPool1d(1, stride=8,ceil_mode=False)
        
        self.layer1 = self.down_layer(self.input_dim, self.layer_n, self.kernel_size,1, 2)
        self.layer2 = self.down_layer(self.layer_n, int(self.layer_n*2), self.kernel_size,2, 2)
        self.layer3 = self.down_layer(int(self.layer_n*2)+int(self.input_dim), int(self.layer_n*3), self.kernel_size,2, 2)
        self.layer4 = self.down_layer(int(self.layer_n*3)+int(self.input_dim), int(self.layer_n*4), self.kernel_size,2, 2)
        self.layer5 = self.down_layer(int(self.layer_n*4)+int(self.input_dim), int(self.layer_n*5), self.kernel_size,2, 2)

        self.cbr_up1 = conbr_block(int(self.layer_n*7), int(self.layer_n*3), self.kernel_size, 1, 1)
        self.cbr_up2 = conbr_block(int(self.layer_n*5), int(self.layer_n*2), self.kernel_size, 1, 1)
        self.cbr_up3 = conbr_block(int(self.layer_n*3), self.layer_n, self.kernel_size, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.outcov = nn.Conv1d(self.layer_n, 64, kernel_size=self.kernel_size, stride=1,padding = 3)

        self.sdtw = soft_dtw.SoftDTW(True, gamma=5, normalize=False, bandwidth=0.1)
        self.dtw = dtw.DTW(True, normalize=False, bandwidth=1)
        
    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(re_block(out_layer,out_layer,kernel,1))
        return nn.Sequential(*block)
            
    def forward(self, x):
        y = x.clone()
        x = x.permute(0,2,1)
        x = x[:,:,:(x.shape[2]//8)*8]

        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)
        pool_x3 = self.AvgPool1D3(x)
        
        #############Encoder#####################
        
        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)
        
        x = torch.cat([out_1,pool_x1],1)
        out_2 = self.layer3(x)
        
        x = torch.cat([out_2,pool_x2],1)
        x = self.layer4(x)
        
        #############Decoder####################
        
        up = self.upsample1(x)
        up = torch.cat([up,out_2],1)
        up = self.cbr_up1(up)
        
        up = self.upsample(up)
        up = torch.cat([up,out_1],1)
        up = self.cbr_up2(up)
        
        up = self.upsample(up)
        up = torch.cat([up,out_0],1)
        up = self.cbr_up3(up)
        
        out = self.outcov(up)
        
        #out = nn.functional.softmax(out,dim=2)
        out = out.permute(0,2,1)
        return out
    
    def _get_epoch(self):
        references = ['u0001_g_0000v08.pt','u0002_g_0001v14.pt','u0003_g_0002v17.pt','u0004_g_0003v09.pt','u0005_g_0004v16.pt','u0006_g_0005v14.pt','u0007_g_0006v10.pt','u0008_g_0007v11.pt','u0009_g_0008v23.pt','u0010_g_0009v14.pt','u0011_g_0010v07.pt','u0012_g_0011v14.pt','u0013_g_0012v07.pt','u0014_g_0013v01.pt','u0015_g_0014v08.pt','u0016_g_0015v15.pt','u0017_g_0016v08.pt','u0018_g_0017v08.pt','u0019_g_0018v01.pt','u0020_g_0019v08.pt','u0021_g_0020v08.pt','u0022_g_0021v11.pt','u0023_g_0022v13.pt','u0024_g_0023v01.pt','u0025_g_0024v22.pt','u0026_g_0025v14.pt','u0027_g_0026v16.pt','u0028_g_0027v16.pt','u0029_g_0028v07.pt','u0030_g_0029v10.pt','u0031_g_0030v23.pt','u0032_g_0031v09.pt','u0033_g_0032v24.pt','u0034_g_0033v15.pt','u0035_g_0034v19.pt','u0036_g_0035v11.pt','u0037_g_0036v17.pt','u0038_g_0037v13.pt','u0039_g_0038v21.pt','u0040_g_0039v07.pt','u0041_g_0040v01.pt','u0042_g_0041v17.pt','u0043_g_0042v19.pt','u0044_g_0043v23.pt','u0045_g_0044v06.pt','u0046_g_0045v08.pt','u0047_g_0046v24.pt','u0048_g_0047v03.pt','u0049_g_0048v20.pt','u0050_g_0049v08.pt','u0051_g_0050v07.pt','u0052_g_0051v12.pt','u0053_g_0052v18.pt','u0054_g_0053v06.pt','u0055_g_0054v15.pt','u0056_g_0055v13.pt','u0057_g_0056v23.pt','u0058_g_0057v12.pt','u0059_g_0058v05.pt','u0060_g_0059v16.pt','u0061_g_0060v16.pt','u0062_g_0061v11.pt','u0063_g_0062v07.pt','u0064_g_0063v16.pt','u0065_g_0064v14.pt','u0066_g_0065v13.pt','u0067_g_0066v18.pt','u0068_g_0067v02.pt','u0069_g_0068v24.pt','u0070_g_0069v16.pt','u0071_g_0070v10.pt','u0072_g_0071v05.pt','u0073_g_0072v16.pt','u0074_g_0073v12.pt','u0075_g_0074v09.pt','u0076_g_0075v07.pt','u0077_g_0076v12.pt','u0078_g_0077v12.pt','u0079_g_0078v22.pt','u0080_g_0079v23.pt','u0081_g_0080v03.pt','u0082_g_0081v14.pt','u0083_g_0082v03.pt','u0084_g_0083v17.pt','u0085_g_0084v01.pt','u0086_g_0085v11.pt','u0087_g_0086v04.pt','u0088_g_0087v10.pt','u0089_g_0088v18.pt','u0090_g_0089v03.pt','u0091_g_0090v22.pt','u0092_g_0091v11.pt','u0093_g_0092v19.pt','u0094_g_0093v06.pt','u0095_g_0094v10.pt','u0096_g_0095v20.pt','u0097_g_0096v05.pt','u0098_g_0097v20.pt','u0099_g_0098v10.pt','u0100_g_0099v10.pt']
        random.shuffle(references)

        epoch = []

        for file in references:
            file_prefix = file.split("v")[0]
            for i in range(0,25):
                new_file = file_prefix + 'v{:02d}'.format(i) + '.pt'
                if new_file != file_prefix:
                    epoch.append((new_file, file))

        return epoch

    def _get_batch(self, epoch, batch_size):
        assert (len(epoch)) % batch_size == 0
        batch = epoch[:batch_size]
        epoch = epoch[batch_size:]
        
        return batch, epoch
        
    def _load_tensor(self, tensor_name, features_path = None):
        if features_path is not None:
            return torch.load(os.path.join(features_path,tensor_name))
         
        return torch.load(os.path.join(dataset_folder,tensor_name))
        
    def _load_batch(self, batch):
        with torch.no_grad():
            loaded_batch = [torch.load(os.path.join(dataset_folder,batch[0][1]))]

            for tuple in batch:
                query = None
                query = self._load_tensor(tuple[0])

                loaded_batch.append(query)

        return loaded_batch

    def _dtr(self,x, y, len_x, len_y):
        return self.sdtw(x[None, :int(len_x)], y[None, :int(len_y)])[0]/(len_x + len_y)

        # xx = self.sdtw(x[None, :int(len_x)], x[None, :int(len_x)]).detach().cpu().numpy()[0]/(len_x + len_x)
        # yy = self.sdtw(y[None, :int(len_y)], y[None, :int(len_y)]).detach().cpu().numpy()[0]/(len_y + len_y)
        # xy = self.sdtw(x[None, :int(len_x)], y[None, :int(len_y)]).detach().cpu().numpy()[0]/(len_x + len_y)

        # return xy-0.5*(xx+yy)
    
    def _dte(self, x, y, shape1, shape2):
        # Your dte calculation logic here
        return self.dtw(x[None,], y[None,])[0].detach().cpu().numpy()[0] / (64*(shape1 + shape2))


    def _inference(self, files : str, features_path : str):
        """ Calcula score de dissimilaridade entre um grupo de assinaturas

        Args:
            files (str): nome dos arquivos separados por espaço seguido da label (0: original, 1: falsificada)
            distances (Dict[str, Dict[str, float]]): ex: distances[a][b] = distância entre assinatura a e b.
                distances[a] -> dicionário em que cada chave representa a distância entre a e a chave em questão.
                Esse dicionário contém todas as combinações possíveis entre chaves.
                distances[a][b] == distances[b][a].
        Raises:
            ValueError: caso o formato dos arquivos seja diferente do 4vs1 ou 1vs1.

        Returns:
            Tuple[float, str, int]: score de dissimilaridade, usuário de referência, predição (default: math.nan)
        """
        tokens = files.split(" ")
        user_key = tokens[0].split("_")[0]
        # user_key = tokens[0]

        result = math.nan
        refs = []
        sign = ""

        if len(tokens) == 3: result = int(tokens[2]); refs.append(tokens[0]); sign = tokens[1]
        elif len(tokens) == 6: result = int(tokens[5]); refs = tokens[0:4]; sign = tokens[4]
        else: raise ValueError("Arquivos de comparação com formato desconhecido")

        s_avg = 0
        s_min = 0
        
        dists_query = []
        refs_dists = []
        
        # Obtém a distância entre todas as referências:
        # Isto é: (r1,r2), (r1,r3), (r1,r4), (r2,r3), (r2,r4) e (r3,r4); Obs: o DTW é simétrico, dtw(a,b)==dtw(b,a)
        with torch.no_grad():
            query = self._load_tensor(sign.split('.')[0] + '.pt')
            for i in range(0,len(refs)):
                x = self._load_tensor(refs[i].split('.')[0] + '.pt', features_path).cuda()
                x = torch.unsqueeze(x, dim=0)
                x = self(x)
                x = x.squeeze()

                dists_query.append(self._dtr(x, query, x.shape[0], query.shape[0]))

                for j in range(i+1, len(refs)):
                    y = self._load_tensor(refs[j].split('.')[0] + '.pt', features_path).cuda()
                    y = torch.unsqueeze(y, dim=0)
                    y = self(y)
                    y = y.squeeze()

                    refs_dists.append(self._dtr(x,y,x.shape[0],y.shape[0]))

        refs_dists = np.array(refs_dists)
        dists_query = np.array(dists_query)
        

        """Cálculos de dissimilaridade a partir daqui"""
        dk = 1
        if len(refs_dists) > 1:
            dk = np.mean(refs_dists)
        dk_sqrt = dk ** (1.0/2.0) 

        s_avg = np.mean(dists_query)/dk_sqrt
        s_min = min(dists_query)/dk_sqrt

        return (s_avg + s_min), user_key, result

    def evaluate(self, comparison_file : str, n_epoch : int, result_folder : str, features_path : str):
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

        os.makedirs(result_folder, exist_ok=True)

        file_name = (comparison_file.split(os.sep)[-1]).split('.')[0]
        print("\n\tAvaliando " + file_name)
        comparison_folder = result_folder + os.sep + file_name
        if not os.path.exists(comparison_folder): os.mkdir(comparison_folder)

        users = {}

        for line in tqdm(lines, "Calculando distâncias..."):
            distance, user_id, true_label = self._inference(line, features_path=features_path)
            
            if user_id not in users: 
                users[user_id] = {"distances": [distance], "true_label": [true_label], "predicted_label": []}
            else:
                users[user_id]["distances"].append(distance)
                users[user_id]["true_label"].append(true_label)

        # Nesse ponto, todas as comparações foram feitas
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
                eer, eer_threshold = get_eer(y_true=users[user]["true_label"], y_scores=users[user]["distances"])
                th_range_local = np.max(np.array(users[user]["distances"])[np.array(users[user]["distances"]) < eer_threshold])

                local_ths.append(eer_threshold)
                eers.append(eer)
                local_buffer += user + ", " + "{:.5f}".format(eer) + ", " + "{:.5f}".format(eer_threshold) + ", 0, 0, 0, " + "{:.5f}".format(eer_threshold -th_range_local) + " (" + "{:.5f}".format(th_range_local) + "~" + "{:.5f}".format(eer_threshold) + ")\n"

        print("Obtendo EER global...")
        
        # Calculo do EER global
        eer_global, eer_threshold_global = get_eer(global_true_label, global_distances, result_folder=comparison_folder, generate_graph=True, n_epoch=n_epoch)

        local_eer_mean = np.mean(np.array(eers))
        local_ths = np.array(local_ths)
        local_ths_var  = np.var(local_ths)
        local_ths_amp  = np.max(local_ths) - np.min(local_ths)
        
        th_range_global = np.max(np.array(global_distances)[np.array(global_distances) < eer_threshold_global])

        buffer += "Global, " + "{:.5f}".format(eer_global) + ", " + "{:.5f}".format(eer_threshold_global) + ", " + "{:.5f}".format(local_eer_mean) + ", " + "{:.5f}".format(local_ths_var) + ", " + "{:.5f}".format(local_ths_amp) + ", " + "{:.5f}".format(eer_threshold_global -th_range_global) + " (" + "{:.5f}".format(th_range_global) + "~" + "{:.5f}".format(eer_threshold_global) + ")\n" + local_buffer

        with open(comparison_folder + os.sep + file_name + " epoch=" + str(n_epoch) + ".csv", "w") as fw:
            fw.write(buffer)

        ret_metrics = {"Global EER": eer_global, "Mean Local EER": local_eer_mean, "Global Threshold": eer_threshold_global, "Local Threshold Variance": local_ths_var, "Local Threshold Amplitude": local_ths_amp}
        print (ret_metrics)

        self.train(mode=True)
        return ret_metrics
    
    def start_train(self):
        bckp_path = os.path.join(hyperparameters['test_name'],"Backup")
        os.makedirs(bckp_path, exist_ok=True)

        features_path = os.path.join(hyperparameters['test_name'],"features")
        os.makedirs(features_path, exist_ok=True)

        ## Hyperparameter
        batch_size = 25
        epoch_size = 2500

        ## Build tensor data for torch
        #Build model, initial weight and optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr = hyperparameters['learning_rate'],weight_decay=hyperparameters['decay']) # Using Adam optimizer
        
        self.train()
        avg_loss = 0
        losses = []
        plot_losses = []

        for e in range(1, hyperparameters['epochs']+1):
            epoch = self._get_epoch()
            pbar = tqdm(total=(epoch_size//batch_size), position=0, leave=True, desc="Epoch " + str(e) +" PAL: " + "{:.4f}".format(avg_loss))
            
            while epoch != []:
                batch_names, epoch = self._get_batch(epoch=epoch, batch_size=batch_size)
                batch = self._load_batch(batch_names)
                ref = batch[0].cuda()

                loss = 0

                for j in range(len(batch[1:])):
                    embedding = batch[j].cuda()
                    embedding = torch.unsqueeze(embedding, dim=0)
                    embedding = self(embedding)
                    embedding = embedding.squeeze()

                    ref_emb = self._dtr(ref, embedding, ref.shape[0], embedding.shape[0])
                    contrastive_loss = F.relu(ref_emb - hyperparameters['margin'])
                    
                    loss += contrastive_loss
                    loss += ref_emb
                    if e % hyperparameters['eval_step'] == 0:
                        with torch.no_grad():
                            torch.save(embedding, os.path.join(features_path, batch_names[j][0].split('.')[0] + '.pt'))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                pbar.update(1)
            pbar.close()
            avg_loss = np.mean(np.array(losses))
            plot_losses.append(avg_loss)

            if e % hyperparameters['eval_step'] == 0:
                self.evaluate(MCYT_SKILLED_1VS1, e, result_folder=hyperparameters['test_name'], features_path=features_path)
                self.evaluate(MCYT_SKILLED_4VS1, e, result_folder=hyperparameters['test_name'], features_path=features_path)
                self.evaluate(MCYT_RANDOM_4VS1, e, result_folder=hyperparameters['test_name'], features_path=features_path)
                self.evaluate(MCYT_RANDOM_1VS1, e, result_folder=hyperparameters['test_name'], features_path=features_path)
            
            if len(plot_losses) > hyperparameters['early_stop']:
                if avg_loss >= np.mean(np.array(plot_losses[-10:])):
                    print("*** Early Stop! ***")
                    self.evaluate(MCYT_SKILLED_4VS1, e, result_folder=hyperparameters['test_name'], features_path=features_path)
                    self.evaluate(MCYT_SKILLED_1VS1, e, result_folder=hyperparameters['test_name'], features_path=features_path)
                    self.evaluate(MCYT_RANDOM_4VS1, e, result_folder=hyperparameters['test_name'], features_path=features_path)
                    self.evaluate(MCYT_RANDOM_4VS1, e, result_folder=hyperparameters['test_name'], features_path=features_path)
                    break
            
            torch.save(self.state_dict(), bckp_path + os.sep + "epoch" + '{:04d}'.format(e) + ".pt")

        # Loss graph
        plt.xlabel("#Epoch")
        plt.ylabel("Loss")
        plt.plot(list(range(0,len(plot_losses))), plot_losses)
        plt.savefig(hyperparameters['test_name'] + os.sep + "loss.png")
        plt.cla()
        plt.clf()



parser = argparse.ArgumentParser()
# parser.add_argument("-t", "--test_name", help="set test name", required=True, type=str)
parser.add_argument("-t", "--test_name", help="set test name", required=True, type=str)
parser.add_argument("-w", "--weight", help="set weight name to evaluation", default="epoch2170.pt", type=str)
parser.add_argument("-ep", "--epochs", help="set number of epochs to train the model", default=10000, type=int)
parser.add_argument("-lr", "--learning_rate", help="set learning rate value", default=0.001, type=float)
parser.add_argument("-m", "--margin", help="set margin value", default=1.0, type=float)
parser.add_argument("-dc", "--decay", help="learning rate decay value", default=1e-5, type=float)
parser.add_argument("-es", "--eval_step", help="evaluation step during training and testing all weights", default=50, type=int)
parser.add_argument("-stop", "--early_stop", help="minimum epoch to occur early stop", default=300, type=int)
parser.add_argument("-seed", "--seed", help="set seed value", default=333, type=int)
parser.add_argument("-ev", "--evaluate", help="eavaluate", action='store_true')
args = parser.parse_args()
hyperparameters = vars(args)

if hyperparameters['seed'] is not None:
    random.seed(hyperparameters['seed'])
    np.random.seed(hyperparameters['seed'])
    torch.manual_seed(hyperparameters['seed'])
    torch.cuda.manual_seed(hyperparameters['seed'])
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True

if hyperparameters['evaluate']:
    model = UNET_1D(64,128,7,3) #(input_dim, hidden_layer, kernel_size, depth)
    model = model.to(device)
    f = hyperparameters['test_name'] + os.sep + 'Backup' + os.sep + hyperparameters['weight']
    features_path = os.path.join(hyperparameters['test_name'],"features")
    model.load_state_dict(torch.load(f))
    model.evaluate(MCYT_SKILLED_1VS1, 0, result_folder=hyperparameters['test_name'], features_path=features_path)
    model.evaluate(MCYT_SKILLED_4VS1, 0, result_folder=hyperparameters['test_name'], features_path=features_path)
    model.evaluate(MCYT_RANDOM_4VS1, 0, result_folder=hyperparameters['test_name'], features_path=features_path)
    model.evaluate(MCYT_RANDOM_1VS1, 0, result_folder=hyperparameters['test_name'], features_path=features_path)
    exit()

model = UNET_1D(64,128,7,3) #(input_dim, hidden_layer, kernel_size, depth)
model = model.to(device)
model.start_train()