from ds_pipeline import DsPipeline
from utils.constants import *
import utils.test_protocols as test_protocols
import os
import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
import utils.baysean_search as baysean_search
import utils.parse_arguments as parse_arguments
import wandb

def sweep_train():
	global id
	hyperparameters['wandb_name'] = hyperparameters["test_name"] + ("%03d" % id)
	model = DsPipeline(hyperparameters=hyperparameters)
	print(test_protocols.count_parameters(model))
	model.cuda()
	model.train(mode=True)
	# model.start_train(comparison_files=[SKILLED_STYLUS_4VS1], result_folder=res_folder)
	model.start_train(comparison_files=[SKILLED_STYLUS_4VS1], result_folder=res_folder)
	del model
	id += 1


if __name__ == '__main__':
	# inicializa o parser
	torch.autograd.set_detect_anomaly(True)
	hyperparameters = parse_arguments.parse_arguments()

	print(hyperparameters['test_name'])
	res_folder = hyperparameters['parent_folder'] + os.sep + hyperparameters['test_name'] 
	
	# if not os.path.exists(hyperparameters['parent_folder']): os.mkdir(hyperparameters['parent_folder'])
	# if not os.path.exists(res_folder): os.mkdir(res_folder)
	os.makedirs(res_folder, exist_ok=True)

	if hyperparameters['random_search']:
		test_protocols.random_search_parameters(hyperparameters=hyperparameters)
		exit(0)

	if hyperparameters['baysean_search']:
		baysean_search.baysean_search(hyperparameters['parent_folder'], hyperparameters=hyperparameters)
		exit(0)

	# Sementes e algoritmos deterministicos
	if hyperparameters['seed'] is not None:
		random.seed(hyperparameters['seed'])
		np.random.seed(hyperparameters['seed'])
		torch.manual_seed(hyperparameters['seed'])
		torch.cuda.manual_seed(hyperparameters['seed'])
		print("Using seed " + str(hyperparameters['seed']))
	
	cudnn.enabled = True
	cudnn.benchmark = False
	cudnn.deterministic = True

	if hyperparameters['all_weights']:
		test_protocols.eval_all_weights_stylus(hyperparameters, res_folder)
		exit(0)
	
	if hyperparameters['validate']:
		test_protocols.validation(hyperparameters, res_folder, "1vs1", "stylus")
		test_protocols.validation(hyperparameters, res_folder, "4vs1", "stylus")
		exit(0)

	if hyperparameters['evaluate']:
		test_protocols.evaluate(hyperparameters, res_folder)
		exit(0)

	if hyperparameters['tune_model']:
		model = DsPipeline(hyperparameters=hyperparameters)
		model.cuda()
		model.load_state_dict(torch.load(hyperparameters['weight']))
		
		model.train(mode=True)
		model.start_train(comparison_files=[EBIOSIGN_W1_SKILLED_4VS1], result_folder=res_folder)
		del model
		exit(0)
	
	if hyperparameters['extract_features'] != "":
		result_folder = "Investigation" + os.sep + "Extracted Features" + os.sep + hyperparameters['extract_features']
		f = res_folder + os.sep + 'Backup' + os.sep + hyperparameters['weight']
		model = DsPipeline(hyperparameters=hyperparameters)
		model.cuda()
		model.load_state_dict(torch.load(f))
		model.train(mode=False)
		model.eval()
		model.extract(result_folder=result_folder)
		del model
		exit(0)

	if hyperparameters['cluster']:
		f = res_folder + os.sep + 'Backup' + os.sep + hyperparameters['weight']
		result_folder = res_folder + os.sep + "Clusterized"
		model = DsPipeline(hyperparameters=hyperparameters)
		model.cuda()
		model.train(mode=False)
		model.eval()
		model.load_state_dict(torch.load(f))
		model.clusterize(result_folder=result_folder, comparison_file="pairs.txt")
		
		del model
		exit(0)
	
	"""Experimental"""

	if hyperparameters['sweep']:
		sweep_config = {
			'method': 'bayes',
			"early_terminate": {
				"type": "hyperband",
				"min_iter": 3,
				"eta": 3,
			}
		}
		metric = {
		'name': 'Global EER',
		'goal': 'minimize'   
		}
		sweep_config['metric'] = metric

		parameters_dict = {
			'learning_rate': {
				'distribution': 'uniform',
				'min': 0.003,
				'max': 0.012
			},
			'decay': {
				'distribution': 'uniform',
				'min': 0.6,
				'max': 0.99
			},
			'momentum': {
				'distribution': 'uniform',
				'min': 0.6,
				'max': 0.99
			},
			'random_margin': {
				'distribution': 'uniform',
				'min': 1.2,
				'max': 4
			},
		}
		sweep_config['parameters'] = parameters_dict

		sweep_id = wandb.sweep(sweep_config, project=hyperparameters['wandb_project_name'])
		id = 1
		if hyperparameters['sweep_id'] == "":
			wandb.agent(sweep_id, sweep_train, count=hyperparameters['wandb_number_of_tests'])
		else:
			wandb.agent(hyperparameters['sweep_id'], sweep_train, count=hyperparameters['wandb_number_of_tests'])
		exit(0)

	# if hyperparameters['knn_generate_matrix']:
	# 	f = res_folder + os.sep + 'Backup' + os.sep + hyperparameters['weight']
	# 	model = DsPipeline(hyperparameters=hyperparameters)
	# 	model.cuda()
	# 	model.load_state_dict(torch.load(f))
		
	# 	model.train(mode=False)
	# 	model.eval()
	# 	model.knn_generate_matrix(result_folder=res_folder)
	# 	del model
	# 	exit(0)
 
	# if hyperparameters['knn']:
	# 	f = res_folder + os.sep + 'Backup' + os.sep + hyperparameters['weight']
	# 	model = DsPipeline(hyperparameters=hyperparameters)
	# 	model.cuda()
	# 	model.load_state_dict(torch.load(f))
		
	# 	model.train(mode=False)
	# 	model.eval()
	# 	model.knn(matrix_path=hyperparameters['knn_matrix'], comparison_file=MCYT_SKILLED_4VS1, result_folder=res_folder, n_epoch=10000)
	# 	del model
	# 	exit(0)
 
	# if hyperparameters['transfer_learning']:
	# 	teacher_model = DsPipeline(hyperparameters=hyperparameters)
	# 	teacher_model.cuda()
	# 	teacher_model.load_state_dict(torch.load(hyperparameters['weight']))

	# 	# Treinamento
	# 	model = DsPipeline(hyperparameters=hyperparameters)
	# 	print(test_protocols.count_parameters(model))
	# 	model.cuda()
	# 	model.train(mode=True)
	# 	model.start_transfer(comparison_files=[SKILLED_STYLUS_4VS1], result_folder=res_folder, teacher_model=teacher_model)
	# 	del teacher_model
	# 	del model
	# 	exit(0)
	"""Experimental"""

	# Treinamento
	model = DsPipeline(hyperparameters=hyperparameters)
	print(test_protocols.count_parameters(model))
	model.cuda()
	model.train(mode=True)
	# model.start_train(comparison_files=[SKILLED_STYLUS_4VS1], result_folder=res_folder)
	model.start_train(comparison_files=[RANDOM_STYLUS_4VS1, SKILLED_STYLUS_4VS1], result_folder=res_folder)
	del model