import torch
import torch.nn as nn

from tqdm import tqdm
import os
import random

import DTW.soft_dtw_cuda as soft_dtw
import numpy as np
from matplotlib import pyplot as plt

device = torch.device("cuda")
# dataset_folder = os.path.join("..","Resultados", "ROT_X2_", "ROT_X2_005", "generated_features")
dataset_folder = os.path.join("ROT_X2_", "ROT_X2_005", "generated_features")
training_guide = "training_guide.txt"

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
        
    def _load_batch(self, batch):
        with torch.no_grad():
            loaded_batch = [torch.load(os.path.join(dataset_folder,batch[0][1]))]

            for tuple in batch:
                query = None
                query = torch.load(os.path.join(dataset_folder,tuple[0]))

                loaded_batch.append(query)

        return loaded_batch


    def _dtr(self,x, y, len_x, len_y):
        return self.sdtw(x[None, :int(len_x)], y[None, :int(len_y)])[0]/(len_x + len_y)

    def start_train(self, n_epochs : int = 10000):
        bckp_path = os.path.join("GenerateReference","Backup")
        os.makedirs(bckp_path, exist_ok=True)

        ## Hyperparameter
        lr = 0.001
        batch_size = 25
        epoch_size = 2500

        ## Build tensor data for torch
        #Build model, initial weight and optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr = lr,weight_decay=1e-5) # Using Adam optimizer
        
        self.train()
        avg_loss = 0
        losses = []
        plot_losses = []

        for e in range(n_epochs):
            epoch = self._get_epoch()
            pbar = tqdm(total=(epoch_size//batch_size), position=0, leave=True, desc="Epoch " + str(e) +" PAL: " + "{:.4f}".format(avg_loss))
            
            while epoch != []:
                batch, epoch = self._get_batch(epoch=epoch, batch_size=batch_size)
                batch = self._load_batch(batch)
                ref = batch[0].cuda()

                loss = 0

                ref_ref = self._dtr(ref, ref, ref.shape[0], ref.shape[0])
                for j in range(len(batch[1:])):
                    embedding = batch[j].cuda()
                    embedding = torch.unsqueeze(embedding, dim=0)
                    embedding = self(embedding)
                    embedding = embedding.squeeze()

                    emd_emb = self._dtr(embedding, embedding, embedding.shape[0], embedding.shape[0])
                    ref_emb = self._dtr(ref, embedding, ref.shape[0], embedding.shape[0])
                    loss += (ref_emb - 0.5*(emd_emb+ref_ref))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                pbar.update(1)
            pbar.close()
            avg_loss = np.mean(np.array(losses))
            plot_losses.append(avg_loss)
            
            if len(plot_losses) > 200:
                if avg_loss > np.min(np.array(plot_losses[-10:])):
                    print("Early Stop!")
                    break
            
            torch.save(self.state_dict(), bckp_path + os.sep + "epoch" + + "{:.4f}".format(e) + ".pt")

        # Loss graph
        plt.xlabel("#Epoch")
        plt.ylabel("Loss")
        plt.plot(list(range(0,len(plot_losses))), self.loss_variation)
        plt.savefig("GenerateReference" + os.sep + "loss.png")
        plt.cla()
        plt.clf()

model = UNET_1D(64,128,7,3) #(input_dim, hidden_layer, kernel_size, depth)
model = model.to(device)
model.start_train()