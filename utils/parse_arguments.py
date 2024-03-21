import argparse    
import math
import os

def parse_arguments():
    # Initialize parser
	parser = argparse.ArgumentParser()
	
	#Network
	parser.add_argument("-nout", "--nout", help="number of 1st convolution layer filters", default=64, type=int)
	parser.add_argument("-nh", "--nhidden", help="number of 2nd convolutional layer filters", default=128, type=int)
	# Optimization
	parser.add_argument("-lt", "--loss_type", help="choose loss type (triplet_loss, triplet_mmd, compact_triplet_mmd)", type=str, default="compact_triplet_mmd")
	parser.add_argument("-lr", "--learning_rate", help="set learning rate value", default=0.01, type=float)
	parser.add_argument("-mm", "--momentum", help="set SGD momentum value", default=0.9, type=float)
	parser.add_argument("-dc", "--decay", help="learning rate decay value", default=0.9, type=float)
	parser.add_argument("-stop", "--early_stop", help="minimum epoch to occur early stop", default=26, type=int)
	# results
	parser.add_argument("-df", "--dataset_folder", help="set dataset folder", default=".." + os.sep + "Data" + os.sep + "DeepSignDB", type=str)
	parser.add_argument("-t", "--test_name", help="set test name", required=True, type=str)
	parser.add_argument("-pf", "--parent_folder", help="set folder where test will be saved.", type=str, default="Resultados")
	parser.add_argument("-dsc", "--dataset_scenario", help="stylus, finger or mix", type=str, default="stylus")
	parser.add_argument("-w", "--weight", help="name of weight to be used in evaluation", type=str, default="best.pt")
	parser.add_argument("-es", "--eval_step", help="evaluation step during training and testing all weights", default=3, type=int)
	parser.add_argument("-nt", "--number_of_tests", help="number of search tests (-sg)", default=10, type=int)
	parser.add_argument("-wdb", "--wandb", help="Turns on wandb", action="store_true")
	parser.add_argument("-wpn", "--wandb_project_name", help="Wandb name for training mode", default="", type=str)
	# general parameters
	parser.add_argument("-bs", "--batch_size", help="set batch size (should be dividible by 64)", default=64, type=int)
	parser.add_argument("-ep", "--epochs", help="set number of epochs to train the model", default=25, type=int)
	parser.add_argument("-fdtw", "--use_fdtw", help="use fast dtw on evaluation", action='store_true')
	parser.add_argument("-z", "--zscore", help="normalize x and y coordinates using zscore", action='store_true')
	parser.add_argument("-seed", "--seed", help="set seed value", default=None, type=int)
	parser.add_argument("-ng", "--ng", help="number of genuine signatures in a mini-batch", default=5, type=int)
	parser.add_argument("-nf", "--nf", help="number of forgery signatures in a mini-batch", default=10, type=int)
	parser.add_argument("-nw", "--nw", help="number of writers in a batch", default=4, type=int)
	parser.add_argument("-trans", "--transfer_learning", help="transfer learning method for distillation loss", action='store_true')
	parser.add_argument("-tune", "--tune_model", help="tune model initializing it with a specific weight", action='store_true')
	parser.add_argument("-pa", "--pre_alignment", help="use dtw pre alignment during pre processing", action='store_true')
	parser.add_argument("-rot", "--rotation", help="max rotation augmentation angle, 0.0 if disabled", action='store_true')
	parser.add_argument("-nsl", "--number_of_slices", help="number of writers in a batch", default=5, type=int)
	parser.add_argument("-sps", "--suppress", help="suppress zero pression on data loader", action='store_true')
	# loss hyperparameters
	parser.add_argument("-ga", "--ga", help="turns on gradient for alpha.", action="store_true")
	parser.add_argument("-a", "--alpha", help="set alpha value for icnn_loss or positive signatures variance for triplet loss.", default=math.nan, type=float)
	parser.add_argument("-b", "--beta", help="set beta value for variance of negative signatures", default=math.nan, type=float)
	parser.add_argument("-p", "--p", help="set p value for icnn_loss", default=math.nan, type=float)
	parser.add_argument("-q", "--q", help="set p value for icnn_loss", default=math.nan, type=float)
	parser.add_argument("-r", "--r", help="set r value for icnn_loss", default=math.nan, type=float)
	parser.add_argument("-mkn", "--mmd_kernel_num", help="MMD Kernel Num", default=5, type=float)
	parser.add_argument("-mkm", "--mmd_kernel_mul", help="MMD Kernel Mul", default=2, type=float)
	parser.add_argument("-lbd", "--model_lambda", help="triplet loss model lambda", default=0.01, type=float)
	parser.add_argument("-tm", "--margin", help="triplet loss margin", default=1.0, type=float)
	parser.add_argument("-mmax", "--margin_max", help="set max margin value for distillation loss", default=1.0, type=float)
	parser.add_argument("-mmin", "--margin_min", help="set min margin value for distillation loss", default=0.5, type=float)
	# Testing
	parser.add_argument("-aw", "--all_weights", help="eval all weights", action='store_true')
	parser.add_argument("-ev", "--evaluate", help="validate model using best weights", action='store_true')
	parser.add_argument("-val", "--validate", help="evaluate all mini datasets in -scene + -mode", action='store_true')
	parser.add_argument("-rs", "--random_search", help="search hyperparameters in a random way", action='store_true')
	parser.add_argument("-bays", "--baysean_search", help="search hyperparameters with a baysean search", action='store_true')

	# Read arguments from command line
	args = parser.parse_args()
	hyperparameters = vars(args)

	if hyperparameters["wandb"] == True:
		hyperparameters["wandb_name"] = hyperparameters["test_name"]
	else:
		hyperparameters["wandb_name"] = None

	return hyperparameters