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
import torch.backends.cudnn as cudnn
CHEAT = True
import warnings

from ds_dtw_pytorch_gpu1 import DsDTW
import pandas as pd
import loader
warnings.filterwarnings("ignore")

TEST = "th_test28garu"
RESULT_FOLDER = "thPredict" + os.sep + TEST
DEVELOPMENT = True
mode = 'g'
import random
# mode = 's'

if not os.path.exists(RESULT_FOLDER): os.mkdir(RESULT_FOLDER)

out_file = "predictions_lr" 

class DsPredict(nn.Module):
    def __init__(self, batch_size : int, in_channels : int, dataset_folder : str, lr : float = 0.01):
        super(DsPredict, self).__init__()

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

        self.loss = nn.MSELoss()
    
        self.linear = nn.Linear(self.n_hidden, 16, bias=False)
        self.fc1 = nn.Linear(32, 128, bias=False)
        self.fc2 = nn.Linear(128, 64, bias=False)
        self.fc3 = nn.Linear(64, 1, bias=False)
        self.relu = nn.LeakyReLU()
        self.abs_difs = []
        self.losses = []

        # nn.init.kaiming_normal_(self.linear.weight, a=1)
        # nn.init.kaiming_normal_(self.linear2.weight, a=1) 
        # nn.init.kaiming_normal_(self.cran[0].weight, a=0)
        # nn.init.kaiming_normal_(self.cran[3].weight, a=0)
        # nn.init.zeros_(self.cran[0].bias)
        # nn.init.zeros_(self.cran[3].bias)

        self.rnn = nn.GRU(16, 32, 2, dropout=0.1, batch_first=False, bidirectional=False) #(input_size,hidden_size,num_layers)
        ## close update gate
        for i in range(2):
            eval("self.rnn.bias_hh_l%d"%i)[32:2*32].data.fill_(-1e10) #Initial update gate bias
            eval("self.rnn.bias_ih_l%d"%i)[32:2*32].data.fill_(-1e10) #Initial update gate bias
        
        self.new_sdtw_fw = dtw_cuda.DTW(True, normalize=False, bandwidth=1)
        # self.new_sdtw_fw = new_soft_dtw.SoftDTW(True, gamma=5, normalize=False, bandwidth=1)
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
    
    def forward(self, x):
        # length = torch.sum(mask, dim=1)

        h = self.cran(x.unsqueeze(0).transpose(1,2)).squeeze(0).transpose(0,1)
        # h = self.bn(h, length.int())
        
        # h = h.transpose(1,2)
        # h = h * mask.unsqueeze(2)
        
        h = self.enc1(src=h)
            # h = self.enc2(src=h, src_key_padding_mask=(~mask.bool()))

        h = self.linear(h)

        h,_ = self.rnn(h)

        # if self.training:
        #     return F.avg_pool1d(h.permute(0,2,1),2,2,ceil_mode=False).permute(0,2,1), (length//2).float()

        h = self.fc1(h)
        h = self.relu(h)
        h = self.fc2(h)
        h = self.relu(h)
        h = self.fc3(h)
        h = self.relu(h)

        return torch.mean(h)
    
    def evaluation(self, test, users, labels, epoch_n):
        self.train(False)
        self.eval()

        abs_difs = 0
        den = 0

        buffer = "file, predicted, real, difference\n"
        for user_id in tqdm(test, desc="Avaliando..."):
            den += len(users['u' + f"{user_id:04}" + mode])
            for file in users['u' + f"{user_id:04}" + mode]:
                features = loader.get_features(file, 'stylus', development=DEVELOPMENT)

                inputs = Variable(torch.from_numpy(features)).cuda().transpose(0,1)
                label = Variable(torch.tensor(labels['u' + f"{user_id:04}" + mode])).cuda()

                res = self(inputs.float())

                abs_difs += pow(abs(res.item()-label.item()), 2)
                buffer += file + ", " + str(res.item()) + ", " + str(label.item()) + ", " + str(abs(res.item()-label.item())) + "\n"

        with open(RESULT_FOLDER + os.sep + out_file + "_epoch" + str(epoch_n) + ".csv", "w") as fw:
            fw.write(buffer)

        self.abs_difs.append(abs_difs)

        print("Erro acumulado: " + "{:.4f}".format(abs_difs))
        print("Erro medio acumulado: " + "{:.4f}".format(abs_difs/den))

        self.train(True)
    
BATCH_SIZE = 16
FEATURES = [0,1,2,3,4,5,6,7,8,9,10,11]
# FEATURES=[0,1,2]
DATASET_FOLDER = "Data" + os.sep + "DeepSignDB"
N_EPOCHS = 100
GAMMA = 5
PARENT_FOLDER = "ds_test187"
ITERATION = 2
LEARNING_RATE = 0.01

FILE = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_DeepSignDB_skilled_stylus_4vs1.txt"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def load_labels(file_name = "thPredict/labels.csv"):
    df = pd.read_csv(file_name, sep=', ', header=None, skiprows=2, names=["user", "eer_local", "th_local"])   
    users = df["user"]
    th_local = df["th_local"]

    labels = {}

    for i in range(0, len(users)):
        labels[users[i] + mode] = th_local[i]

    return labels

if __name__ == '__main__':
    # eval_all_scenarios()

    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True

    model1 = DsDTW(batch_size=BATCH_SIZE, in_channels=len(FEATURES), dataset_folder=DATASET_FOLDER, gamma=5)
    model1.load_state_dict(torch.load(PARENT_FOLDER + os.sep + "Backup" + os.sep + "best.pt"))
    # model.cuda()
    # model.train(mode=False)
    # model.eval()
    # model.new_evaluate(FILE, 60000, result_folder=PARENT_FOLDER)

    
    model2 = DsPredict(batch_size=BATCH_SIZE, in_channels=len(FEATURES), dataset_folder=DATASET_FOLDER)
    
    # with torch.no_grad():
    model2.linear.load_state_dict(model1.linear.state_dict())
    model2.enc1.load_state_dict(model1.enc1.state_dict())
    model2.cran.load_state_dict(model1.cran.state_dict())

    model2.linear.requires_grad_(False)
    model2.enc1.requires_grad_(False)
    model2.cran.requires_grad_(False)
        
        

    """"""
    model = model2

    model.cuda()

    print(count_parameters(model))
    path = "Data" + os.sep + "DeepSignDB" + os.sep + "Development" + os.sep + "stylus"
    users = batches_gen.get_files(path)
    num_epochs = 10

    labels = load_labels("thPredict/labels.csv")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  

    train = random.sample(list(range(1, 499)) + list(range(1009,1085)), 402)
    test = list(set(list(range(1, 499)) + list(range(1009,1085))) - set(train))

    rl = 0
    den = 1
    for i in range(1, N_EPOCHS+1):
        running_loss = 0
        for user_id in tqdm(train, desc="Epoch: " + str(i) + " Loss: " + "{:.4f}".format(rl/den)):
            if i == 1: den+= len(users['u' + f"{user_id:04}" + mode])
            for file in users['u' + f"{user_id:04}" + mode]:

                features = loader.get_features(file, 'stylus', development=DEVELOPMENT)

                inputs = Variable(torch.from_numpy(features), requires_grad=True).cuda().transpose(0,1)
                label = Variable(torch.tensor(labels['u' + f"{user_id:04}" + mode]), requires_grad=True).cuda()

                res = model(inputs.float())
                l = model.loss(res.float(), label.float())
                running_loss += l.item()

                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            
        model.evaluation(test, users, labels, i)
        model.losses.append(running_loss/den)

        rl = running_loss

    # Loss graph
    plt.xlabel("#Epoch")
    plt.ylabel("Loss")
    plt.plot(list(range(1,len(model.losses)+1)), model.losses)
    plt.savefig(RESULT_FOLDER + os.sep + "loss.png")
    plt.cla()
    plt.clf()  

     # Loss graph
    plt.xlabel("#Epoch")
    plt.ylabel("Erro quadrático acumulado")
    plt.plot(list(range(1,len(model.abs_difs)+1)), model.abs_difs)
    plt.savefig(RESULT_FOLDER + os.sep + "abs_difs.png")
    plt.cla()
    plt.clf() 