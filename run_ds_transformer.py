from ds_transformer import DsTransformer
import os
import gc
import inspect
import torch
import torch.backends.cudnn as cudnn
import argparse

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
    parser.add_argument("-tl", "--triplet_loss_w", help="set triplet loss weight", default=0.7, type=float)

    # Read arguments from command line
    args = parser.parse_args()

    PARENT_FOLDER = "Resultados" + os.sep + "ds_test277"
    print(args.test_name)


    # eval_all_scenarios()

    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    res_folder = "Resultados" + os.sep + args.test_name

    if not args.evaluate:
        """Iniciar treino"""
        model = DsTransformer(batch_size=args.batch_size, in_channels=len(args.features), dataset_folder=args.dataset_folder, gamma=args.gamma, lr=args.learning_rate)
        print(count_parameters(model))
        model.cuda()
        model.train(mode=True)
        model.start_train(n_epochs=args.epochs, batch_size=args.batch_size, comparison_files=[FILE], result_folder=res_folder, triplet_loss_w=args.triplet_loss_w)
    else:
        """Avaliar modelo"""
        model = DsTransformer(batch_size=args.batch_size, in_channels=len(args.features), dataset_folder=args.dataset_folder, gamma=args.gamma, lr=args.learning_rate)
        print(count_parameters(model))
        model.load_state_dict(torch.load(res_folder + os.sep + "Backup" + os.sep + "best.pt"))
        model.cuda()
        model.train(mode=False)
        model.eval()
        model.new_evaluate(FILE, 0, result_folder=res_folder)

    """Continuar treino""" 
    # model = DsTransformer(batch_size=BATCH_SIZE, in_channels=len(FEATURES), dataset_folder=DATASET_FOLDER, gamma=5)
    # model.load_state_dict(torch.load(res_folder + os.sep + "Backup" + os.sep + "epoch50.pt"))
    # model.cuda()
    # model.train(mode=True)
    # model.start_train(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, comparison_files=[FILE], result_folder=res_folder+'_'+str(ITERATION))

    