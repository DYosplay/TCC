import numpy as np
from typing import List, Tuple
import os
from matplotlib import pyplot as plt
from shapely.geometry import LineString
from sklearn.metrics import roc_curve
import numpy.typing as npt


def generate_graph(legit : List[float], forgery : List[float], epoch, result_folder : str, user : str = '0'):
        total_distances = np.array(legit + forgery)
        total_distances = np.sort(total_distances)

        frr_list = []
        far_list = []

        for dist in total_distances:
            frr = np.sum(legit >= dist) / len(legit)
            frr_list.append(frr)
            
            far = np.sum(forgery < dist) / len(forgery)
            far_list.append(far)

        frr_list = np.array(frr_list)
        far_list = np.array(far_list)

        if not os.path.exists(result_folder + os.sep + user):
            os.mkdir(result_folder + os.sep + user)

        plt.plot(total_distances, frr_list, 'b', label="FRR")
        plt.plot(total_distances, far_list, 'r', label="FAR")
        plt.legend(loc="upper right")

        line_1 = LineString(np.column_stack((total_distances, frr_list)))
        line_2 = LineString(np.column_stack((total_distances, far_list)))
        intersection = line_1.intersection(line_2)
        x,y = intersection.xy
        
        plt.xlabel("Threshold")
        plt.ylabel("Error Rate")
        plt.plot(*intersection.xy, 'ro')
        plt.text(x[0]+0.05,y[0]+0.05, "EER = " + "{:.3f}".format(y[0]))
        plt.savefig(result_folder + os.sep + user + os.sep + "Epoch" + str(epoch) + ".png")
        plt.cla()
        plt.clf()

        return y[0], x[0]

def get_multidimensional_eer(users, legit : npt.ArrayLike, forgery : npt.ArrayLike, n_epoch : int = 0, result_folder : str = None, user : str = '0', generate_graph : bool = False):
    """ Calculate EER using n thresholds. 

    Args:
        legit (npt.ArrayLike[npt.ArrayLike]): Distances of genuine signatures. Each element is a np.array containing n values and is related with one genuine signature.
        forgery (npt.ArrayLike[npt.ArrayLike]): Distances of forgeries. Each element is a np.array containing n values and is related with one forgery signature.
        epoch (_type_): epoch number
        result_folder (str): where the graph should be stored
        user (str, optional): user id. Defaults to '0'.

    Returns:
        (float, npt.ArrayLike): eer, optimal threshold
    """
    legit = np.array(legit)
    forgery = np.array(forgery)
    total_distances = np.concatenate((legit, forgery), axis=0)
    total_distances = np.sort(total_distances, axis=0)

    from sklearn.manifold import TSNE
    from matplotlib import pyplot as plt
    from sklearn.datasets import load_iris
    from numpy import reshape
    import seaborn as sns
    import pandas as pd

    y = 0
    for key in users.keys():
        x = np.array(users[key]['distances'])
        z = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(x)     
        scikit = pd.DataFrame()
        scikit["y"] = y
        y+=1
        scikit[str(y) + "comp-1"] = z[:,0]
        scikit[str(y) + "comp-2"] = z[:,1]
    sns.scatterplot().set (title="Scikit learn TSNE") 
    plt.show()
    frr_list = []
    far_list = []

    for dist in total_distances:
    # for i in range(0, len(total_distances)):
    #     for j in range(0, len(total_distances[0])):
             

        aux_l = (legit < dist)
        frr = np.sum(np.logical_not(np.apply_along_axis(np.all, axis=1, arr=aux_l)).astype(int)) / len(legit)
        frr_list.append(frr)
        
        aux_f = (forgery < dist)
        far = np.sum(np.apply_along_axis(np.all, axis=1, arr=aux_f).astype(int)) / len(forgery)
        far_list.append(far)

    frr_list = np.array(frr_list)
    far_list = np.array(far_list)
    
    indexes = np.array(list(range(len(total_distances))))

    crossing_index = np.where(far_list >= frr_list)[0][0]

    # Interpolate the exact crossing value
    crossing_value = np.interp(crossing_index, [crossing_index - 1, crossing_index],
                            [far_list[crossing_index - 1], far_list[crossing_index]])

    print("Crossing Index:", crossing_index)
    print("Crossing Value:", crossing_value)

    line_1 = LineString(np.column_stack((indexes, frr_list)))
    line_2 = LineString(np.column_stack((indexes, far_list)))
    intersection = line_1.intersection(line_2)
    x,y = intersection.xy

    if generate_graph:
        if not os.path.exists(result_folder + os.sep + user):
            os.mkdir(result_folder + os.sep + user)

        plt.plot(indexes, frr_list, 'b', label="FRR")
        plt.plot(indexes, far_list, 'r', label="FAR")
        plt.legend(loc="upper right")
        
        plt.xlabel("Threshold")
        plt.ylabel("Error Rate")
        plt.plot(*intersection.xy, 'ro')
        plt.text(x[0]+0.05,y[0]+0.05, "EER = " + "{:.3f}".format(y[0]))
        plt.savefig(result_folder + os.sep + user + os.sep + "Epoch" + str(n_epoch) + ".png")
        plt.cla()
        plt.clf()

    return y[0], x[0]

def get_eer(y_true = List[int], y_scores = List[float], result_folder : str = None, generate_graph : bool = False, n_epoch : int = None) -> Tuple[float, float]:
        fpr, tpr, threshold = roc_curve(y_true=y_true, y_score=y_scores, pos_label=1)
        fnr = 1 - tpr

        far = fpr
        frr = fnr

        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # as a sanity check the value should be close to
        eer2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

        eer = (eer + eer2)/2
        # eer = min(eer, eer2)

        if generate_graph:
            frr_list = np.array(frr)
            far_list = np.array(far)

            plt.plot(threshold, frr_list, 'b', label="FRR")
            plt.plot(threshold, far_list, 'r', label="FAR")
            plt.legend(loc="upper right")
 
            plt.xlabel("Threshold")
            plt.ylabel("Error Rate")
            plt.plot(eer_threshold, eer, 'ro')
            plt.text(eer_threshold + 0.05, eer+0.05, s="EER = " + "{:.5f}".format(eer))
            #plt.text(eer_threshold + 1.05, eer2+1.05, s="EER = " + "{:.5f}".format(eer2))
            plt.savefig(result_folder + os.sep + "Epoch" + str(n_epoch) + ".png")
            plt.cla()
            plt.clf()

        return eer, eer_threshold