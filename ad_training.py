import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import random
import numpy as np
from ds_transformer import DsTransformer
from adda_discriminator import ADDA_Discriminator
import torch.optim as optim
from torch.autograd import Variable
from utils import mmd_loss as mmd

from tqdm import tqdm

import DataLoader.batches_gen as batches_gen

RESULT_FOLDER = "Resultados"
if not os.path.exists(RESULT_FOLDER): os.mkdir(RESULT_FOLDER)

FILE = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_DeepSignDB_skilled_stylus_4vs1.txt"

parser = argparse.ArgumentParser()

parser.add_argument("-lr", "--learning_rate", help="set learning rate value", default=0.001, type=float)
parser.add_argument("-df", "--dataset_folder", help="set dataset folder", default=".." + os.sep + "Data" + os.sep + "DeepSignDB", type=str)
parser.add_argument("-g", "--gamma", help="set gamma value for soft-dtw", default=5, type=int)
parser.add_argument("-bs", "--batch_size", help="set batch size (should be dividible by 16)", default=16, type=int)
parser.add_argument("-f", "--features", help="list of index of features used by the model", default=[0,1,2,3,4,5,6,7,8,9,10,11], type=list)
parser.add_argument("-ep", "--epochs", help="set number of epochs to train the model", default=30, type=int)
parser.add_argument("-t", "--test_name", help="set name of current test", type=str, required=True)
parser.add_argument("-lf", "--load_folder", help="set name of current test", type=str, required=True)
parser.add_argument("-fdtw", "--use_fdtw", help="use fast dtw on evaluation", action='store_true')
parser.add_argument("-lt", "--loss_type", help="choose loss type (triplet_loss, cosface, arcface, sphereface, icnn_loss, quadruplet_loss, triplet_mmd, triplet_coral, norm_triplet_mmd)", type=str, default='triplet_loss')
parser.add_argument("-a", "--alpha", help="set alpha value for icnn_loss or positive signatures variance for triplet loss.", default=0.0, type=float)
parser.add_argument("-b", "--beta", help="set beta value for variance of negative signatures", default=0.0, type=float)
parser.add_argument("-p", "--p", help="set p value for icnn_loss", default=0.0, type=float)
parser.add_argument("-q", "--q", help="set q value for icnn_loss", default=0.0, type=float)
parser.add_argument("-r", "--r", help="set r value for icnn_loss", default=0.0, type=float)
parser.add_argument("-seed", "--seed", help="set seed value", default=111, type=int)
parser.add_argument("-tm", "--margin", help="set margin value for triplet loss margin", default=1.0, type=float)
parser.add_argument("-dc", "--decay", help="learning rate decay value", default=0.9, type=float)
parser.add_argument("-stop", "--early_stop", help="set number of epoch which enables early stop", default=26, type=int)

parser.add_argument("-w", "--weight", help="name of weight to be used in evaluation", type=str, default="best.pt")
parser.add_argument("-mode", "--mode", help="stylus or finger", type=str, default="stylus")
parser.add_argument("-scene", "--scenario", help="4vs1 or 1vs1", type=str, default="4vs1")

parser.add_argument("-z", "--zscore", help="normalize x and y coordinates using zscore", action='store_true')

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Read arguments from command line
args = parser.parse_args()

print(args.test_name)

# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
# cudnn.enabled = True
# cudnn.benchmark = False
# cudnn.deterministic = True

load_folder = "Resultados" + os.sep + args.load_folder
result_folder = "Resultados" + os.sep + args.test_name

# Inicializar o modelo de source
# python ad_training.py -ep=25 -t=ds_triplet_mmd_333 -lr 0.01 -bs 32 -lt=triplet_mmd -a 1.0 -tm 1 -p 0.9 -q 0.1 -dc 0.9 -stop 26
source_encoder = DsTransformer(batch_size=args.batch_size, in_channels=len(args.features), dataset_folder=args.dataset_folder, gamma=args.gamma, lr=args.learning_rate, loss_type=args.loss_type, alpha=args.alpha, beta=args.beta, p=args.p, q=args.q, r=args.r, decay = args.decay, use_fdtw = args.use_fdtw)
print(count_parameters(source_encoder))
source_encoder.load_state_dict(torch.load(load_folder + os.sep + "Backup" + os.sep + args.weight))
for p in source_encoder.parameters():
    p.requires_grad = False
print(count_parameters(source_encoder))
source_encoder.cuda()

# Treinar o modelo de target junto com o discriminator
target_encoder = DsTransformer(batch_size=args.batch_size, in_channels=len(args.features), dataset_folder=args.dataset_folder, gamma=args.gamma, lr=args.learning_rate, loss_type=args.loss_type, alpha=args.alpha, beta=args.beta, p=args.p, q=args.q, r=args.r, decay = args.decay, use_fdtw = args.use_fdtw)
target_encoder.load_state_dict(torch.load(load_folder + os.sep + "Backup" + os.sep + args.weight))
print(count_parameters(target_encoder))
target_encoder.cuda()

discriminator = ADDA_Discriminator()
discriminator.cuda()
# loss_bce = nn.BCELoss()
loss_bce = mmd.MMDLoss()

target_optimizer = optim.Adam(params=target_encoder.parameters(), lr=args.learning_rate)
disc_optimizer = optim.Adam(params=discriminator.parameters(), lr=args.learning_rate)

num_disc_steps_per_epoch = 2
num_tar_steps_per_epoch = 2

if not os.path.exists(result_folder):
    os.mkdir(result_folder)

if not os.path.exists(result_folder + os.sep + "Backup_disc"):
    os.mkdir(result_folder + os.sep + "Backup_disc")
bckp_path_disc = result_folder + os.sep + "Backup_disc"

if not os.path.exists(result_folder + os.sep + "Backup"):
    os.mkdir(result_folder + os.sep + "Backup")
bckp_path_target = result_folder + os.sep + "Backup"

running_loss_disc = 0
losses_disc = []

running_loss_trg = 0
losses_trg = []

for i in range(1, args.epochs+1):

    epoch = batches_gen.ad_generate_epoch()
    epoch_aux = epoch.copy()

    epoch_size = len(epoch)
    scenario = 'stylus'
    
    for j in range(1, num_disc_steps_per_epoch+1):     
        loss_value_disc = running_loss_disc/epoch_size
        losses_disc.append(loss_value_disc)

        print("Discriminador rodada " + str(j))

        pbar = tqdm(total=(epoch_size//(args.batch_size//16)), position=0, leave=True, desc="Epoch " + str(i) +" PAL: " + "{:.3f}".format(loss_value_disc))

        while epoch != []:
            src_batch, src_lens, trg_batch, trg_lens, epoch = batches_gen.ad_get_batch_from_epoch(epoch, args.batch_size, z=args.zscore, scenario=scenario) 

            src_mask = source_encoder.getOutputMask(src_lens)
            src_mask = Variable(torch.from_numpy(src_mask)).cuda()
            src_inputs = Variable(torch.from_numpy(src_batch)).cuda()

            outputs_source, length_source = source_encoder(src_inputs.float(), src_mask, i)

            trg_mask = target_encoder.getOutputMask(trg_lens)
            trg_mask = Variable(torch.from_numpy(trg_mask)).cuda()
            trg_inputs = Variable(torch.from_numpy(trg_batch)).cuda()
            
            outputs_target, length_target = target_encoder(trg_inputs.float(), trg_mask, i)
            
            pred_source = discriminator(outputs_source)
            pred_target = discriminator(outputs_target)

            label_source = torch.ones(pred_source.size()).cuda()
            label_target = torch.zeros(pred_target.size()).cuda()

            src_loss = loss_bce(pred_source, label_source)
            trg_loss = loss_bce(pred_target, label_target)

            loss_disc = src_loss + trg_loss

            disc_optimizer.zero_grad()
            loss_disc.backward()
            disc_optimizer.step()
            running_loss_disc += loss_disc.item()

            torch.save(discriminator.state_dict(), bckp_path_disc + os.sep + "epoch" + str(i) + ".pt")

            pbar.update(1)

        pbar.close()

    epoch = epoch_aux

    for j in range(1, num_disc_steps_per_epoch+1):     
        loss_value_trg = running_loss_trg/epoch_size
        losses_trg.append(loss_value_trg)

        print("Target rodada " + str(j))

        pbar = tqdm(total=(epoch_size//(args.batch_size//16)), position=0, leave=True, desc="Epoch " + str(i) +" PAL: " + "{:.3f}".format(loss_value_trg))

        while epoch != []:
            src_batch, src_lens, trg_batch, trg_lens, epoch = batches_gen.ad_get_batch_from_epoch(epoch, args.batch_size, z=args.zscore, scenario=scenario) 
        
            trg_mask = target_encoder.getOutputMask(trg_lens)
            trg_mask = Variable(torch.from_numpy(trg_mask)).cuda()
            trg_inputs = Variable(torch.from_numpy(trg_batch)).cuda()
            
            outputs_target, length_target = target_encoder(trg_inputs.float(), trg_mask, i)

            pred_target = discriminator(outputs_target)
            label_target = torch.ones(pred_target.size()).cuda()

            loss_trg = loss_bce(pred_target,label_target)

            target_optimizer.zero_grad()
            loss_trg.backward()
            target_optimizer.step()

            running_loss_trg += loss_trg.item()

            torch.save(discriminator.state_dict(), bckp_path_target + os.sep + "epoch" + str(i) + ".pt")

            pbar.update(1)

        pbar.close()

    if i % 3 == 0 or i == 1:
        target_encoder.new_evaluate(FILE, i, result_folder)