from ds_dtw_pytorch_gpu1 import DsDTW
import os
import gc
import inspect
import torch
import torch.backends.cudnn as cudnn
BATCH_SIZE = 16
FEATURES = [0,1,2,3,4,5,6,7,8,9,10,11]
# FEATURES=[0,1,2]
DATASET_FOLDER = "Data" + os.sep + "DeepSignDB"
N_EPOCHS = 30
GAMMA = 5
PARENT_FOLDER = "ds_test243"
LEARNING_RATE = 0.001

FILE = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_DeepSignDB_skilled_stylus_4vs1.txt"
FILE8 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "1vs1" + os.sep + "skilled" + os.sep + "Comp_DeepSignDB_skilled_stylus_1vs1.txt"
FILE9 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "random" + os.sep + "Comp_DeepSignDB_random_stylus_4vs1.txt"
FILE10 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "1vs1" + os.sep + "random" + os.sep + "Comp_DeepSignDB_random_stylus_1vs1.txt"


FILE_FINGER1 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "finger" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_DeepSignDB_skilled_finger_4vs1.txt"
FILE_FINGER2 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "finger" + os.sep + "1vs1" + os.sep + "skilled" + os.sep + "Comp_DeepSignDB_skilled_finger_1vs1.txt"
FILE_FINGER3 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "finger" + os.sep + "4vs1" + os.sep + "random" + os.sep + "Comp_DeepSignDB_random_finger_4vs1.txt"
FILE_FINGER4 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "finger" + os.sep + "1vs1" + os.sep + "random" + os.sep + "Comp_DeepSignDB_random_finger_1vs1.txt"


FILE2 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_eBioSignDS1_W1_skilled_stylus_4vs1.txt"
FILE3 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_eBioSignDS1_W2_skilled_stylus_4vs1.txt"
FILE4 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_eBioSignDS1_W3_skilled_stylus_4vs1.txt"
FILE5 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_eBioSignDS1_W4_skilled_stylus_4vs1.txt"
FILE6 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_eBioSignDS1_W5_skilled_stylus_4vs1.txt"
FILE7 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "jprotocol.txt"



PATH = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal"

def validation(model : DsDTW):
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
        model = DsDTW(batch_size=BATCH_SIZE, in_channels=len(FEATURES), gamma=gamma, dataset_folder=DATASET_FOLDER)
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
        model = DsDTW(batch_size=BATCH_SIZE, in_channels=len(FEATURES), gamma=gammas[i], dataset_folder=DATASET_FOLDER)
        model.load_state_dict(torch.load(res_folder + os.sep + "Backup" + os.sep + "epoch" + str(best_w[i]) + ".pt"))
        model.cuda()

        model.train(mode=False)
        model.eval()

        for file in eval_f:
            model.new_evaluate(file, 400, result_folder=res_folder)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # eval_all_scenarios()

    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True

    res_folder = PARENT_FOLDER + "_gamma_" + str(GAMMA)
    model = DsDTW(batch_size=BATCH_SIZE, in_channels=len(FEATURES), dataset_folder=DATASET_FOLDER, gamma=GAMMA, lr=LEARNING_RATE)
    # model = torch.compile(model)
    # model.load_state_dict(torch.load(res_folder + os.sep + "Backup" + os.sep + "best.pt"))
    print(count_parameters(model))

    model.cuda()
    model.train(mode=True)
    model.start_train(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, comparison_files=[FILE], result_folder=res_folder)
    # model.start_train(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, comparison_files=[FILE], result_folder=PARENT_FOLDER)
    # model = DsDTW(batch_size=BATCH_SIZE, in_channels=len(FEATURES), dataset_folder=DATASET_FOLDER, gamma=5)
    # model.load_state_dict(torch.load(res_folder + os.sep + "Backup" + os.sep + "best.pt"))
    # model.cuda()
    # model = torch.compile(model)

    # model.train(mode=False)
    # model.eval()


    # eval_all_weights(model)
    # validation(model)

    # model.new_evaluate(FILE_FINGER1, 119, result_folder=PARENT_FOLDER)
    # model.new_evaluate(FILE3, 119, result_folder=PARENT_FOLDER)
    # model.new_evaluate(FILE4, 119, result_folder=PARENT_FOLDER)
    # model.new_evaluate(FILE5, 119, result_folder=PARENT_FOLDER)
    # model.new_evaluate(FILE6, 119, result_folder=PARENT_FOLDER)

    # model.new_evaluate(FILE3, 300, result_folder=res_folder)
    # model.new_evaluate(FILE8, 100, result_folder=PARENT_FOLDER)
    # model.new_evaluate(FILE9, 2, result_folder=PARENT_FOLDER)
    # model.new_evaluate(FILE10, 2, result_folder=PARENT_FOLDER)

    # model.new_evaluate(FILE7, 0, result_folder=PARENT_FOLDER)
    # model.evaluate(comparions_files=[FILE2], n_epoch=100, result_folder=PARENT_FOLDER)
    # model.evaluate(comparions_files=[FILE], n_epoch=100, result_folder=PARENT_FOLDER)
    # model.evaluate(comparions_files=[FILE3], n_epoch=25, result_folder=PARENT_FOLDER)
    # model.evaluate(comparions_files=[FILE4], n_epoch=25, result_folder=PARENT_FOLDER)
    # model.evaluate(comparions_files=[FILE5], n_epoch=25, result_folder=PARENT_FOLDER)
    # model.evaluate(comparions_files=[FILE6], n_epoch=25, result_folder=PARENT_FOLDER)

        