from ds_transformer import DsTransformer
import os
import gc
import inspect
import torch
import torch.backends.cudnn as cudnn
import argparse
import random
import numpy as np


FILE = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_DeepSignDB_skilled_stylus_4vs1.txt"
FILE8 = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "1vs1" + os.sep + "skilled" + os.sep + "Comp_DeepSignDB_skilled_stylus_1vs1.txt"
FILE9 = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "random" + os.sep + "Comp_DeepSignDB_random_stylus_4vs1.txt"
FILE10 = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "1vs1" + os.sep + "random" + os.sep + "Comp_DeepSignDB_random_stylus_1vs1.txt"

FILE_FINGER1 = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "finger" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_DeepSignDB_skilled_finger_4vs1.txt"
FILE_FINGER2 = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "finger" + os.sep + "1vs1" + os.sep + "skilled" + os.sep + "Comp_DeepSignDB_skilled_finger_1vs1.txt"
FILE_FINGER3 = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "finger" + os.sep + "4vs1" + os.sep + "random" + os.sep + "Comp_DeepSignDB_random_finger_4vs1.txt"
FILE_FINGER4 = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "finger" + os.sep + "1vs1" + os.sep + "random" + os.sep + "Comp_DeepSignDB_random_finger_1vs1.txt"

FILE2 = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_eBioSignDS1_W1_skilled_stylus_4vs1.txt"
FILE3 = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_eBioSignDS1_W2_skilled_stylus_4vs1.txt"
FILE4 = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_eBioSignDS1_W3_skilled_stylus_4vs1.txt"
FILE5 = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_eBioSignDS1_W4_skilled_stylus_4vs1.txt"
FILE6 = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_eBioSignDS1_W5_skilled_stylus_4vs1.txt"
FILE7 = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "jprotocol.txt"

FILE11 = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_MCYT_skilled_stylus_4vs1.txt"


PATH = ".." + os.sep + "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal"

def validation(model : DsTransformer):
    finger_path = PATH + os.sep + "finger"
    stylus_path = PATH + os.sep + "stylus"

    opts = ["1vs1" + os.sep + "random", "4vs1" + os.sep + "random", "1vs1" + os.sep + "skilled", "4vs1" + os.sep + "skilled"]

    if not os.path.exists(PARENT_FOLDER): os.mkdir(PARENT_FOLDER)

    if not os.path.exists(PARENT_FOLDER + os.sep + "stylus"): os.mkdir(PARENT_FOLDER + os.sep + "stylus")

    if not os.path.exists(PARENT_FOLDER + os.sep + "finger"): os.mkdir(PARENT_FOLDER + os.sep + "finger")

    print("\nEvaluating finger scenario")
    
    for opt in opts:
        path = finger_path + os.sep + opt + os.sep
        files = os.listdir(finger_path + os.sep + opt)

        for file in files:
            model.new_evaluate(path + file, n_epoch=777, result_folder=PARENT_FOLDER)

    print("Evaluating stylus scenario")
    
    for opt in opts:
        path = stylus_path + os.sep + opt + os.sep
        files = os.listdir(stylus_path + os.sep + opt)

        for file in files:
            model.new_evaluate(path + file, n_epoch=777, result_folder=PARENT_FOLDER)

    with open(PARENT_FOLDER + os.sep + "log.csv", "w") as fw:
        fw.write(model.buffer)

def validate(model : DsTransformer, res_folder : str):
    stylus_path = PATH + os.sep + "stylus"


    paths = [stylus_path + os.sep + "4vs1" + os.sep + "skilled" + os.sep, stylus_path + os.sep + "1vs1" + os.sep + "skilled" + os.sep]
    paths += [stylus_path + os.sep + "4vs1" + os.sep + "random" + os.sep, stylus_path + os.sep + "1vs1" + os.sep + "random" + os.sep]

    for path in paths:
        files = os.listdir(path)

        for file in files:
            model.new_evaluate(path + file, n_epoch=777, result_folder=res_folder)

        with open(res_folder + os.sep + "log.csv", "w") as fw:
            fw.write(model.buffer)

def eval_all_weights(model):
    if not os.path.exists(PARENT_FOLDER + os.sep + "all_weights"):
        os.mkdir(PARENT_FOLDER + os.sep + "all_weights")

    for i in range(16, N_EPOCHS+1):
        file = PARENT_FOLDER + os.sep + 'Backup' + os.sep + "epoch" + str(i) + ".pt" 
        model.load_state_dict(torch.load(file))
        model.new_evaluate(FILE, 1000+i, result_folder=PARENT_FOLDER)

def free_memory(to_delete: list):
    calling_namespace = inspect.currentframe().f_back

    for _var in to_delete:
        calling_namespace.f_locals.pop(_var, None)
        gc.collect()
        torch.cuda.empty_cache()

def all_scenarios():
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True

    gammas = [0.1, 1, 5, 10]
    eval = [FILE_FINGER1, FILE_FINGER2, FILE_FINGER3, FILE_FINGER4, FILE, FILE8, FILE9, FILE10]
    cf = [FILE]
    for gamma in gammas:
        res_folder = PARENT_FOLDER + "_gamma_" + str(gamma)
        model = DsTransformer(batch_size=BATCH_SIZE, in_channels=len(FEATURES), gamma=gamma, dataset_folder=DATASET_FOLDER)
        print(count_parameters(model))
        model.cuda()
        # model = torch.compile(model)
        model.train(mode=True)
        model.start_train(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, comparison_files=cf, result_folder=res_folder)
        # model = torch.compile(model)

        model.train(mode=False)
        model.eval()

        for file in eval:
            model.new_evaluate(file, 119, result_folder=res_folder)

def eval_all_scenarios():
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True


    best_w = [25,28,20,20]
    gammas = [0.1, 1, 5, 10]
    # eval_f = [FILE_FINGER1, FILE_FINGER2, FILE_FINGER3, FILE_FINGER4, FILE, FILE8, FILE9, FILE10]
    eval_f = [FILE8, FILE_FINGER1, FILE_FINGER2]
    cf = [FILE]
    for i in range(len(gammas)):
        res_folder = PARENT_FOLDER + "_gamma_" + str(gammas[i])
        model = DsTransformer(batch_size=BATCH_SIZE, in_channels=len(FEATURES), gamma=gammas[i], dataset_folder=DATASET_FOLDER)
        model.load_state_dict(torch.load(res_folder + os.sep + "Backup" + os.sep + "epoch" + str(best_w[i]) + ".pt"))
        model.cuda()

        model.train(mode=False)
        model.eval()

        for file in eval_f:
            model.new_evaluate(file, 400, result_folder=res_folder)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def jprotocol():
    model = DsTransformer(batch_size=BATCH_SIZE, in_channels=len(FEATURES), dataset_folder=DATASET_FOLDER, gamma=5)
    model.load_state_dict(torch.load(PARENT_FOLDER + os.sep + "Backup" + os.sep + "best.pt"))
    model.cuda()
    model.train(mode=False)
    model.eval()
    model.new_evaluate(FILE7, 10000, result_folder=PARENT_FOLDER)

if __name__ == '__main__':
    if not os.path.exists("Resultados"): os.mkdir("Resultados")


    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("-lr", "--learning_rate", help="set learning rate value", default=0.001, type=float)
    parser.add_argument("-df", "--dataset_folder", help="set dataset folder", default=".." + os.sep + "Data" + os.sep + "DeepSignDB", type=str)
    parser.add_argument("-g", "--gamma", help="set gamma value for soft-dtw", default=5, type=int)
    parser.add_argument("-bs", "--batch_size", help="set batch size (should be dividible by 16)", default=16, type=int)
    parser.add_argument("-f", "--features", help="list of index of features used by the model", default=[0,1,2,3,4,5,6,7,8,9,10,11], type=list)
    parser.add_argument("-ep", "--epochs", help="set number of epochs to train the model", default=30, type=int)
    # parser.add_argument("-cf", "--comparison_file", help="set the comparison file used in the evaluation during training", default='FILE', type=str)
    parser.add_argument("-t", "--test_name", help="set name of current test", type=str, required=True)
    parser.add_argument("-ev", "--evaluate", help="validate model using best weights", action='store_true')
    parser.add_argument("-ft", "--fine_tuning", help="fine tuning model", action='store_true')
    parser.add_argument("-tl", "--triplet_loss_w", help="set triplet loss weight", default=1.0, type=float)
    parser.add_argument("-m", "--mask", help="set triplet loss weight", action='store_true')
    parser.add_argument("-c", "--compile", help="user model compile (only with torch>=2.0)", action='store_true')
    parser.add_argument("-val", "--validate", help="eval stylus 4vs1 scenario", action='store_true')
    parser.add_argument("-lt", "--loss_type", help="choose loss type (triplet_loss, cosface, arcface, sphereface, icnn_loss, quadruplet_loss, triplet_mmd, triplet_coral, norm_triplet_mmd)", type=str, default='triplet_loss')
    parser.add_argument("-a", "--alpha", help="set alpha value for icnn_loss or positive signatures variance for triplet loss.", default=0.0, type=float)
    parser.add_argument("-b", "--beta", help="set beta value for variance of negative signatures", default=0.0, type=float)
    parser.add_argument("-p", "--p", help="set p value for icnn_loss", default=0.0, type=float)
    parser.add_argument("-q", "--q", help="set q value for icnn_loss", default=0.0, type=float)
    parser.add_argument("-r", "--r", help="set r value for icnn_loss", default=0.0, type=float)
    parser.add_argument("-seed", "--seed", help="set seed value", default=111, type=int)
    parser.add_argument("-qm", "--quadruplet_margin", help="set margin value for quadruplet margin", default=0.5, type=float)
    parser.add_argument("-tm", "--margin", help="set margin value for triplet loss margin", default=1.0, type=float)
    parser.add_argument("-dc", "--decay", help="learning rate decay value", default=0.9, type=float)
    parser.add_argument("-nlr", "--new_learning_rate", help="choose new_learning_rate value", type=float, default=0.001)
    parser.add_argument("-encs", "--encoders", help="set number of transformer encoder layers", default=1, type=int)
    # Read arguments from command line
    args = parser.parse_args()
    
    print(args.test_name)


    # eval_all_scenarios()

    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    res_folder = "Resultados" + os.sep + args.test_name

    if args.fine_tuning: 
        model = DsTransformer(batch_size=args.batch_size, in_channels=len(args.features), dataset_folder=args.dataset_folder, gamma=args.gamma, lr=args.learning_rate, use_mask=args.mask, loss_type=args.loss_type, alpha=args.alpha, beta=args.beta, p=args.p, q=args.q, r=args.r, qm=args.quadruplet_margin, margin = args.margin, decay = args.decay, nlr = args.new_learning_rate, encs = args.encoders)
        if args.compile:
            model = torch.compile(model)
        print(count_parameters(model))
        model.load_state_dict(torch.load(res_folder + os.sep + "Backup" + os.sep + "best.pt"))
        model.cuda()
        model.train(mode=True)
        model.start_train(n_epochs=args.epochs, batch_size=args.batch_size, comparison_files=[FILE11], result_folder=res_folder, triplet_loss_w=args.triplet_loss_w)

    elif not args.evaluate and not args.validate:
        """Iniciar treino"""
        model = DsTransformer(batch_size=args.batch_size, in_channels=len(args.features), dataset_folder=args.dataset_folder, gamma=args.gamma, lr=args.learning_rate, use_mask=args.mask, loss_type=args.loss_type, alpha=args.alpha, beta=args.beta, p=args.p, q=args.q, r=args.r, qm=args.quadruplet_margin, margin = args.margin, decay = args.decay, nlr = args.new_learning_rate, encs = args.encoders)
        if args.compile:
            model = torch.compile(model)
        print(count_parameters(model))
        model.cuda()
        model.train(mode=True)
        model.start_train(n_epochs=args.epochs, batch_size=args.batch_size, comparison_files=[FILE11], result_folder=res_folder, triplet_loss_w=args.triplet_loss_w)
    elif args.evaluate:
        """Avaliar modelo"""
        model = DsTransformer(batch_size=args.batch_size, in_channels=len(args.features), dataset_folder=args.dataset_folder, gamma=args.gamma, lr=args.learning_rate, use_mask=args.mask, loss_type=args.loss_type, alpha=args.alpha, beta=args.beta, p=args.p, q=args.q, r=args.r, qm=args.quadruplet_margin, margin = args.margin, decay = args.decay, nlr = args.new_learning_rate, encs = args.encoders)
        if args.compile:
            model = torch.compile(model)
        print(count_parameters(model))
        model.load_state_dict(torch.load(res_folder + os.sep + "Backup" + os.sep + "best.pt"))
        model.cuda()
        model.train(mode=False)
        model.eval()
        model.new_evaluate(FILE, 777, result_folder=res_folder)
        # model.new_evaluate(FILE10, 777, result_folder=res_folder)
    
    elif args.validate:
        model = DsTransformer(batch_size=args.batch_size, in_channels=len(args.features), dataset_folder=args.dataset_folder, gamma=args.gamma, lr=args.learning_rate, use_mask=args.mask, loss_type=args.loss_type, alpha=args.alpha, beta=args.beta, p=args.p, q=args.q, r=args.r, qm=args.quadruplet_margin, margin = args.margin, decay = args.decay, nlr = args.new_learning_rate)
        if args.compile:
            model = torch.compile(model)
        print(count_parameters(model))
        model.load_state_dict(torch.load(res_folder + os.sep + "Backup" + os.sep + "best.pt"))
        model.cuda()
        model.train(mode=False)
        model.eval()
        validate(model, res_folder)

    """Continuar treino""" 
    # model = DsTransformer(batch_size=BATCH_SIZE, in_channels=len(FEATURES), dataset_folder=DATASET_FOLDER, gamma=5)
    # model.load_state_dict(torch.load(res_folder + os.sep + "Backup" + os.sep + "epoch50.pt"))
    # model.cuda()
    # model.train(mode=True)
    # model.start_train(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, comparison_files=[FILE], result_folder=res_folder+'_'+str(ITERATION))

    