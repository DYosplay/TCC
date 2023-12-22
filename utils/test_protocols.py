import os
from typing import Dict, Any

from ds_pipeline import DsPipeline
from utils.constants import *


import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
import math

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def random_search_parameters(hyperparameters : Dict[str, Any]):
	""" Realiza uma busca aleatória pelos hiperparametros alpha, beta, p e r

        Args:
            search_name (str): nome da bateria de testes
    """
	alpha = np.round(np.random.uniform(0.7,1.2,40),2)
	beta  = np.round(np.random.uniform(0.5,4,40),1)
	p     = np.round(np.random.uniform(0.05,1,40),3)
	r     = np.round(np.random.uniform(0.05,1,40),3)

	random.seed(hyperparameters["seed"])
	np.random.seed(hyperparameters["seed"])
	torch.manual_seed(hyperparameters["seed"])
	torch.cuda.manual_seed(hyperparameters["seed"])

	cudnn.enabled = True
	cudnn.benchmark = False
	cudnn.deterministic = True

	if not os.path.exists(hyperparameters['search_name']): os.mkdir(hyperparameters["search_name"])
    
	parm_log = "Name, alpha, beta, p, r, EER\n"
	for i in range(0, hyperparameters['number_of_tests']):
		try:
			print("Name, alpha, beta, p, r")
			print("ctl_" + f'{i:03d},' + str(alpha[i]) + "," + str(beta[i]) + "," + str(p[i]) + "," + str(r[i]))
			res_folder = hyperparameters['search_name'] + os.sep + "ctl_" + f'{i:03d}'
			
			if not os.path.exists(res_folder): os.mkdir(res_folder)

			hyperparameters["p"] = p[i]
			hyperparameters['r'] = r[i]
			hyperparameters['alpha'] = alpha[i]
			hyperparameters['beta'] = beta[i]
			
			model = DsPipeline(hyperparameters=hyperparameters)

			print("Parâmetros treináveis:" + str(count_parameters(model)))
			
			model.cuda()
			model.train(mode=True)
			model.start_train(comparison_files=[SKILLED_STYLUS_4VS1], result_folder=res_folder)
			
			parm_log += "ctl_" + f'{i:03d},' + str(alpha[i]) + "," + str(beta[i]) + "," + str(p[i]) + "," + str(r[i]) + "," + str(model.best_eer) + "\n" 
		except:
			continue

		with open(hyperparameters['search_name'] + os.sep + "log.csv", "w") as fw:
			fw.write(parm_log)

def eval_all_weights(model : DsPipeline, res_folder : str, file : str, iter : int, n_epochs : int = 25):
	for i in range(1, n_epochs+1):
		f = res_folder + os.sep + 'Backup' + os.sep + "epoch" + str(i) + ".pt" 
		model.load_state_dict(torch.load(f))
		model.new_evaluate(file, iter+i, result_folder=res_folder)

		with open(res_folder + os.sep + file.split(os.sep)[-1] + " summary.csv", "w") as fw: 
			fw.write(model.buffer)
		model.buffer = "Epoch, mean_local_eer, global_eer, th_global, var_th, amp_th\n"
		model.best_eer = math.inf

def eval_all_weights_stylus(model : DsPipeline, res_folder : str, n_epochs : int):
	eval_all_weights(model, res_folder, SKILLED_STYLUS_1VS1, 1000, n_epochs)
	eval_all_weights(model, res_folder, RANDOM_STYLUS_1VS1, 1000, n_epochs)
	eval_all_weights(model, res_folder, SKILLED_STYLUS_4VS1, 1000, n_epochs)
	eval_all_weights(model, res_folder, RANDOM_STYLUS_4VS1, 1000, n_epochs)

def validation(model : DsPipeline, res_folder : str, n_refs : str = "4vs1", mode : str = 'stylus'):
	path = PATH + os.sep + mode
	
	opts = [n_refs + os.sep + "random", n_refs + os.sep + "skilled"]

	if not os.path.exists(res_folder): os.mkdir(res_folder)

	if not os.path.exists(res_folder + os.sep + mode): os.mkdir(res_folder + os.sep + mode)

	print("Evaluating " + mode + " scenario")
	
	for opt in opts:
		p = path + os.sep + opt
		files = os.listdir(p)
		
		for file in files:
			model.new_evaluate(p + os.sep + file, n_epoch=0, result_folder=res_folder + os.sep + mode)

	with open(res_folder + os.sep + "log_" + n_refs + "_" + mode + ".csv" , "w") as fw:
		fw.write(model.buffer)
