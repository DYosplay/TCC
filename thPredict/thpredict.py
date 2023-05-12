import torch
import torch.nn as nn
import numpy as np
import pandas as pd

input_size = 12
hidden_size = 6
num_out = 1
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_out):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_out)  
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = torch.mean(out)
        return out
    
def load_labels(file_name = "labels.csv"):
    df = pd.read_csv(file_name, sep=', ', header=None, skiprows=2, names=["user", "eer_local", "th_local"])   
    users = df["user"]
    th_local = df["th_local"]

    labels = {}

    for i in range(0, len(users)):
        labels[users[i]] = th_local[i]

    return labels

load_labels()    

model = NeuralNet(input_size, hidden_size, num_out).cuda()
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

for i in range(1, num_epochs+1):
    data = np.random.rand(np.random.randint(110, 120), 12)
    label = np.random.random()

    inputs = torch.from_numpy(data).cuda()
    label = torch.tensor(label).cuda()

    res = model(inputs.float())
    l = loss(res, label)
    
    optimizer.zero_grad()
    l.backward()
    optimizer.step()


    


