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
N_EPOCHS = 40
PARENT_FOLDER = "recriar10"

FILE = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_DeepSignDB_skilled_stylus_4vs1.txt"
FILE2 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_eBioSignDS1_W1_skilled_stylus_4vs1.txt"
FILE3 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_eBioSignDS1_W2_skilled_stylus_4vs1.txt"
FILE4 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_eBioSignDS1_W3_skilled_stylus_4vs1.txt"
FILE5 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_eBioSignDS1_W4_skilled_stylus_4vs1.txt"
FILE6 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "skilled" + os.sep + "Comp_eBioSignDS1_W5_skilled_stylus_4vs1.txt"
FILE7 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "jprotocol.txt"

FILE8 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "1vs1" + os.sep + "skilled" + os.sep + "Comp_DeepSignDB_skilled_stylus_1vs1.txt"
FILE9 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "4vs1" + os.sep + "random" + os.sep + "Comp_DeepSignDB_random_stylus_4vs1.txt"
FILE10 = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal" + os.sep + "stylus" + os.sep + "1vs1" + os.sep + "random" + os.sep + "Comp_DeepSignDB_random_stylus_1vs1.txt"

PATH = "Data" + os.sep + "DeepSignDB" + os.sep + "Comparison_Files" + os.sep + "TBIOM_2021_Journal"

def validation(model : DsDTW):
    finger_path = PATH + os.sep + "finger"
    stylus_path = PATH + os.sep + "stylus"

    opts = ["1vs1" + os.sep + "random", "4vs1" + os.sep + "random", "1vs1" + os.sep + "skilled", "4vs1" + os.sep + "skilled"]

    if not os.path.exists(PARENT_FOLDER): os.mkdir(PARENT_FOLDER)

    if not os.path.exists(PARENT_FOLDER + os.sep + "stylus"): os.mkdir(PARENT_FOLDER + os.sep + "stylus")

    if not os.path.exists(PARENT_FOLDER + os.sep + "finger"): os.mkdir(PARENT_FOLDER + os.sep + "finger")

    print("Evaluating stylus scenario")
    
    for opt in opts:
        path = stylus_path + os.sep + opt + os.sep
        files = os.listdir(stylus_path + os.sep + opt)

        for file in files:
            model.new_evaluate(path + file, n_epoch=777, result_folder=PARENT_FOLDER)

    with open(PARENT_FOLDER + os.sep + "log.csv", "w") as fw:
        fw.write(model.buffer)

    # print("\nEvaluating finger scenario")
    
    # for opt in opts:
    #     path = finger_path + os.sep + opt + os.sep
    #     files = os.listdir(finger_path + os.sep + opt)

    #     for file in files:
    #         model.evaluate(comparions_files=[path + file], n_epoch=0, result_folder=PARENT_FOLDER)

def free_memory(to_delete: list):
    calling_namespace = inspect.currentframe().f_back

    for _var in to_delete:
        calling_namespace.f_locals.pop(_var, None)
        gc.collect()
        torch.cuda.empty_cache()
    
if __name__ == '__main__':
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    # if not os.path.exists(PARENT_FOLDER):
    #     os.mkdir(PARENT_FOLDER)

    model = DsDTW(batch_size=BATCH_SIZE, in_channels=len(FEATURES), dataset_folder=DATASET_FOLDER)
    # model = torch.compile(model)
    model.cuda()
    model.train(mode=True)
    model.start_train(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, comparison_files=[FILE], result_folder=PARENT_FOLDER)
    # model.start_train(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, comparison_files=[FILE], result_folder=PARENT_FOLDER)
    # model = DsDTW(batch_size=BATCH_SIZE, in_channels=len(FEATURES), dataset_folder=DATASET_FOLDER)
    # model.load_state_dict(torch.load(PARENT_FOLDER + os.sep + "Backup" + os.sep + "best.pt"))
    # model.cuda()
    # model = torch.compile(model)

    # model.train(mode=False)
    # model.eval()
    # validation(model)

    # model.new_evaluate(FILE2, 119, result_folder=PARENT_FOLDER)
    # model.new_evaluate(FILE3, 119, result_folder=PARENT_FOLDER)
    # model.new_evaluate(FILE4, 119, result_folder=PARENT_FOLDER)
    # model.new_evaluate(FILE5, 119, result_folder=PARENT_FOLDER)
    # model.new_evaluate(FILE6, 119, result_folder=PARENT_FOLDER)

    # model.new_evaluate(FILE, 100, result_folder=PARENT_FOLDER)
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

        