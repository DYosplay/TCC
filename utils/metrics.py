import numpy as np
from typing import List, Tuple
import os
from matplotlib import pyplot as plt
from shapely.geometry import LineString
from sklearn.metrics import roc_curve

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

        plt.plot(total_distances, frr_list, 'r', label="FRR")
        plt.plot(total_distances, far_list, 'b', label="FAR")
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

            plt.plot(threshold, frr_list, 'r', label="FRR")
            plt.plot(threshold, far_list, 'b', label="FAR")
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