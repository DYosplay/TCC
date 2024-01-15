import os
from typing import Dict, Any

from ds_pipeline import DsPipeline
from utils.constants import *


import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def random_search_parameters(hyperparameters : Dict[str, Any]):
	""" Realiza uma busca aleatória pelos hiperparametros alpha, beta, p e r

        Args:
            search_name (str): nome da bateria de testes
    """
	alpha = np.round(np.random.uniform(0.7,1.2,hyperparameters['number_of_tests']),2)
	beta  = np.round(np.random.uniform(0.5,4,hyperparameters['number_of_tests']),1)
	p     = np.round(np.random.uniform(0.05,1,hyperparameters['number_of_tests']),3)
	r     = np.round(np.random.uniform(0.05,1,hyperparameters['number_of_tests']),3)

	random.seed(hyperparameters["seed"])
	np.random.seed(hyperparameters["seed"])
	torch.manual_seed(hyperparameters["seed"])
	torch.cuda.manual_seed(hyperparameters["seed"])

	cudnn.enabled = True
	cudnn.benchmark = False
	cudnn.deterministic = True

	if not os.path.exists(hyperparameters['parent_folder']): os.mkdir(hyperparameters['parent_folder'])
    
	parm_log = "Name, alpha, beta, p, r, EER\n"
	for i in range(0, hyperparameters['number_of_tests']):
		try:
			hyperparameters['test_name'] = hyperparameters['parent_folder'] + "_" + f'{i:03d}'
			if hyperparameters['wandb_name'] is not None: hyperparameters['wandb_name'] = hyperparameters['test_name']

			print("Name, alpha, beta, p, r")
			print(hyperparameters['parent_folder'] + "_" + f'{i:03d},' + str(alpha[i]) + "," + str(beta[i]) + "," + str(p[i]) + "," + str(r[i]))
			res_folder = hyperparameters['parent_folder'] + os.sep + hyperparameters['test_name']
			
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
			# model.start_train(comparison_files=[EBIOSIGN_W1_SKILLED_4VS1], result_folder=res_folder)
			
			parm_log += "ctl_" + f'{i:03d},' + str(alpha[i]) + "," + str(beta[i]) + "," + str(p[i]) + "," + str(r[i]) + "," + str(model.best_eer) + "\n" 
			del model
		except:
			del model
			continue

		with open(hyperparameters['parent_folder'] + os.sep + "log.csv", "w") as fw:
			fw.write(parm_log)

def eval_all_weights(hyperparameters : Dict[str, Any], res_folder : str, file : str, iter : int):
	model = DsPipeline(hyperparameters=hyperparameters)
	model.cuda()
	model.train(mode=False)
	model.eval()
	
	for i in range(1, hyperparameters['epochs']+1):
		f = res_folder + os.sep + 'Backup' + os.sep + "epoch" + str(i) + ".pt" 
		model.load_state_dict(torch.load(f))
		model.new_evaluate(file, iter+i, result_folder=res_folder)

	with open(res_folder + os.sep + file.split(os.sep)[-1] + " summary.csv", "w") as fw: 
		fw.write(model.buffer)

	del model
		

def eval_all_weights_stylus(hyperparameters : Dict[str, Any], res_folder : str):
	if hyperparameters['wandb_name'] is not None: hyperparameters['wandb_name'] = hyperparameters['wandb_name'] + "_AW_SKILLED_STYLUS_1VS1"
	eval_all_weights(hyperparameters, res_folder, SKILLED_STYLUS_1VS1, 1000)
	if hyperparameters['wandb_name'] is not None: hyperparameters['wandb_name'] = hyperparameters['wandb_name'] + "_AW_RANDOM_STYLUS_1VS1"
	eval_all_weights(hyperparameters, res_folder, RANDOM_STYLUS_1VS1, 1000)
	if hyperparameters['wandb_name'] is not None: hyperparameters['wandb_name'] = hyperparameters['wandb_name'] + "_AW_SKILLED_STYLUS_4VS1"
	eval_all_weights(hyperparameters, res_folder, SKILLED_STYLUS_4VS1, 1000)
	if hyperparameters['wandb_name'] is not None: hyperparameters['wandb_name'] = hyperparameters['wandb_name'] + "_AW_RANDOM_STYLUS_4VS1"
	eval_all_weights(hyperparameters, res_folder, RANDOM_STYLUS_4VS1, 1000)

def validate(hyperparameters : Dict[str, Any], res_folder : str, comparison_file : str):
	f = res_folder + os.sep + 'Backup' + os.sep + hyperparameters['weight']
	
	protocol_name = comparison_file.split(os.sep)[-1].split(".")[0]
	if hyperparameters['wandb_name'] is not None: hyperparameters['wandb_name'] = hyperparameters['wandb_name'] + "_VAL_" + protocol_name.upper()

	model = DsPipeline(hyperparameters=hyperparameters)
	model.load_state_dict(torch.load(f))

	model.cuda()
	model.train(mode=False)
	model.eval()

	model.new_evaluate(comparison_file, 0, res_folder)

	log = model.buffer

	del model

	return log

def validation(hyperparameters : Dict[str, Any], res_folder : str, n_refs : str = "4vs1", mode : str = 'stylus'):
	path = PATH + os.sep + mode
	
	opts = [n_refs + os.sep + "random", n_refs + os.sep + "skilled"]

	if not os.path.exists(res_folder): os.mkdir(res_folder)

	if not os.path.exists(res_folder + os.sep + mode): os.mkdir(res_folder + os.sep + mode)

	print("Evaluating " + mode + " scenario")

	log = ""
	
	for opt in opts:
		p = path + os.sep + opt
		files = os.listdir(p)
		
		for file in files:
			log += validate(hyperparameters, res_folder, p + os.sep + file)

	with open(res_folder + os.sep + "log_" + n_refs + "_" + mode + ".csv" , "w") as fw:
		fw.write(log)


def eval_db(hyperparameters : Dict[str, Any], res_folder, comparison_file : str):
	f = res_folder + os.sep + 'Backup' + os.sep + hyperparameters['weight']
	model = DsPipeline(hyperparameters=hyperparameters)
	model.cuda()
	model.train(mode=False)
	model.eval()

	model.load_state_dict(torch.load(f))
	model.new_evaluate(comparison_file, 100, result_folder=res_folder)

	del model

def evaluate(hyperparameters : Dict[str, Any], res_folder):
	# if hyperparameters['wandb_name'] is not None: hyperparameters['wandb_name'] = hyperparameters['wandb_name'] + "_EV_SKILLED_STYLUS_1VS1"
	# eval_db(hyperparameters, res_folder, EBIOSIGN_W1_SKILLED_1VS1)
	if hyperparameters['wandb_name'] is not None: hyperparameters['wandb_name'] = hyperparameters['wandb_name'] + "_EV_SKILLED_STYLUS_1VS1"
	eval_db(hyperparameters, res_folder, SKILLED_STYLUS_1VS1)
	if hyperparameters['wandb_name'] is not None: hyperparameters['wandb_name'] = hyperparameters['wandb_name'] + "_EV_RANDOM_STYLUS_1VS1"
	eval_db(hyperparameters, res_folder, RANDOM_STYLUS_1VS1)
	if hyperparameters['wandb_name'] is not None: hyperparameters['wandb_name'] = hyperparameters['wandb_name'] + "_EV_SKILLED_STYLUS_4VS1"
	eval_db(hyperparameters, res_folder, SKILLED_STYLUS_4VS1)
	if hyperparameters['wandb_name'] is not None: hyperparameters['wandb_name'] = hyperparameters['wandb_name'] + "_EV_RANDOM_STYLUS_4VS1"
	eval_db(hyperparameters, res_folder, RANDOM_STYLUS_4VS1)

