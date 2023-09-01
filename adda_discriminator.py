import torch.nn as nn

class ADDA_Discriminator(nn.Module):
    def __init__(self):
        super(ADDA_Discriminator, self).__init__()
        self.fc1 = nn.Linear(64,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,1)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self ,input):
        out = self.relu(self.fc1(input))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))
        return out