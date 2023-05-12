import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import loader
import os
import batches_gen
from tqdm import tqdm
import random
from torch.autograd import Variable
from matplotlib import pyplot as plt

input_size = 12
hidden_size = 6
num_out = 1

batch_size = 100

num_epochs = 200
learning_rate = 0.0001
TEST = "th_test17_lstm"
RESULT_FOLDER = "thPredict" + os.sep + TEST
DEVELOPMENT = True

if not os.path.exists(RESULT_FOLDER): os.mkdir(RESULT_FOLDER)

out_file = "predictions_lr" + str(learning_rate) 

# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_out):
        super(NeuralNet, self).__init__()
        # MLP
        self.fc01 = nn.Linear(12, 6, bias=True) 
        self.fc02 = nn.Linear(6, 1, bias=True) 
        self.relu = nn.ReLU()

        self.loss = nn.MSELoss()

        # LSTM
        self.lstm = nn.LSTM(12, 32, num_layers=2)
        self.l1 = nn.Linear(32, 1)

        self.losses = []
        self.abs_difs = []
    
    def forward(self, x):
        # MLP
        # out = self.fc01(x)
        # out = self.relu(out)
        # out = self.fc02(out)
        # out = torch.mean(out)

        # LSTM
        out, _ = self.lstm(x)
        out = self.l1(out)
        out = torch.mean(out)
        return out
    
    def evaluation(self, test, users, labels, epoch_n):
        self.train(False)
        self.eval()

        abs_difs = 0
        den = 0

        buffer = "file, predicted, real, difference\n"
        for user_id in tqdm(test, desc="Avaliando..."):
            den += len(users['u' + f"{user_id:04}" + 'g'])
            for file in users['u' + f"{user_id:04}" + 'g']:
                features = loader.get_features(file, 'stylus', development=DEVELOPMENT)

                inputs = Variable(torch.from_numpy(features)).cuda().transpose(0,1)
                label = Variable(torch.tensor(labels['u' + f"{user_id:04}" + 'g'])).cuda()

                res = self(inputs.float())

                abs_difs += pow(abs(res.item()-label.item()), 2)
                buffer += file + ", " + str(res.item()) + ", " + str(label.item()) + ", " + str(abs(res.item()-label.item())) + "\n"

        with open(RESULT_FOLDER + os.sep + out_file + "_epoch" + str(epoch_n) + ".csv", "w") as fw:
            fw.write(buffer)

        self.abs_difs.append(abs_difs)

        print("Erro acumulado: " + "{:.4f}".format(abs_difs))
        print("Erro medio acumulado: " + "{:.4f}".format(abs_difs/den))

        self.train(True)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def load_labels(file_name = "thPredict/labels.csv"):
    df = pd.read_csv(file_name, sep=', ', header=None, skiprows=2, names=["user", "eer_local", "th_local"])   
    users = df["user"]
    th_local = df["th_local"]

    labels = {}

    for i in range(0, len(users)):
        labels[users[i] + 'g'] = th_local[i]

    return labels

model = NeuralNet(input_size, hidden_size, num_out).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

def run(path):
    print(count_parameters(model))
    users = batches_gen.get_files(path)
    labels = load_labels("thPredict/labels_train.csv")

    train = random.sample(list(range(1, 499)) + list(range(1009,1085)), 402)
    test = list(set(list(range(1, 499)) + list(range(1009,1085))) - set(train))

    rl = 0
    den = 1
    for i in range(1, num_epochs+1):
        running_loss = 0
        for user_id in tqdm(train, desc="Epoch: " + str(i) + " Loss: " + "{:.4f}".format(rl/den)):
            if i == 1: den+= len(users['u' + f"{user_id:04}" + 'g'])
            for file in users['u' + f"{user_id:04}" + 'g']:

                features = loader.get_features(file, 'stylus', development=DEVELOPMENT)

                inputs = Variable(torch.from_numpy(features), requires_grad=True).cuda().transpose(0,1)
                label = Variable(torch.tensor(labels['u' + f"{user_id:04}" + 'g']), requires_grad=True).cuda()

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
    plt.ylabel("abs_difs")
    plt.plot(list(range(1,len(model.abs_difs)+1)), model.abs_difs)
    plt.savefig(RESULT_FOLDER + os.sep + "abs_difs.png")
    plt.cla()
    plt.clf()    

# run("Data" + os.sep + "DeepSignDB" + os.sep + "Evaluation" + os.sep + "stylus")
run("Data" + os.sep + "DeepSignDB" + os.sep + "Development" + os.sep + "stylus")


    


    


