    
import numpy as np
from numpy import typing as npt
from Losses import triplet_loss, triplet_mmd, compact_triplet_mmd, clustering_triplet_mmd, contrastive_loss, triplet_loss_offset, compact_triplet_mmd_offset, clustering_triplet_loss, hard_triplet_loss_avg, hard_triplet_loss_max, triplet_distillation, tune_loss, compact_triplet_mmd2
from typing import List, Tuple, Dict, Any
import torch
import json
import os

def define_loss(loss_type : str, ng : int, nf : int, nw : int, margin : torch.nn.Parameter, model_lambda : float, alpha : torch.nn.Parameter, beta : torch.nn.Parameter, p : torch.nn.Parameter, r : torch.nn.Parameter,q : torch.nn.Parameter, mmd_kernel_num : float, mmd_kernel_mul : float, margin_max : float, margin_min : float):
    if loss_type.lower() == "triplet_loss":
        return triplet_loss.Triplet_Loss(ng=ng, nf=nf, nw=nw, margin=margin, model_lambda=model_lambda)
    if loss_type.lower() == "triplet_mmd":
        return triplet_mmd.Triplet_MMD(ng=ng,nf=nf,nw=nw,margin=margin, model_lambda=model_lambda, alpha=alpha, p=p, r=r, mmd_kernel_num=mmd_kernel_num, mmd_kernel_mul=mmd_kernel_mul)
    if loss_type.lower() == "compact_triplet_mmd":
        return compact_triplet_mmd.Compact_Triplet_MMD(ng=ng,nf=nf,nw=nw,margin=margin, alpha=alpha, beta=beta, p=p, r=r, mmd_kernel_num=mmd_kernel_num, mmd_kernel_mul=mmd_kernel_mul)
    if loss_type.lower() == "clustering_triplet_mmd":
        return clustering_triplet_mmd.Clustering_Triplet_MMD(ng=ng,nf=nf,nw=nw,margin=margin, alpha=alpha, beta=beta, p=p, r=r, mmd_kernel_num=mmd_kernel_num, mmd_kernel_mul=mmd_kernel_mul)
    if loss_type.lower() == "contrastive_loss":
        return contrastive_loss.Contrastive_Loss(ng=ng, nf=nf, nw=nw, margin=margin)
    if loss_type.lower() == "triplet_loss_offset":
        return triplet_loss_offset.Triplet_Loss_Offset(ng=ng, nf=nf, nw=nw, margin=margin, model_lambda=model_lambda,alpha=alpha)
    if loss_type.lower() == "clustering_triplet_mmd_offset":
        return compact_triplet_mmd_offset.Compact_Triplet_MMD_Offset(ng=ng,nf=nf,nw=nw,margin=margin, alpha=alpha, beta=beta, p=p, r=r, mmd_kernel_num=mmd_kernel_num, mmd_kernel_mul=mmd_kernel_mul)
    if loss_type.lower() == "clustering_triplet_loss":
        return clustering_triplet_loss.Clustering_Triplet_Loss(ng=ng, nf=nf, nw=nw, margin=margin, model_lambda=model_lambda)
    if loss_type.lower() == "hard_triplet_loss_avg":
        return hard_triplet_loss_avg.Hard_Triplet_Loss(ng=ng, nf=nf, nw=nw, margin=margin, model_lambda=model_lambda)
    if loss_type.lower() == "hard_triplet_loss_max":
        return hard_triplet_loss_max.Hard_Triplet_Loss(ng=ng, nf=nf, nw=nw, margin=margin, model_lambda=model_lambda)
    if loss_type.lower() == "distillation_loss":
        return triplet_distillation.Triplet_Distillation_MMD(ng=ng,nf=nf,nw=nw,margin=margin, alpha=alpha, beta=beta, p=p, r=r, mmd_kernel_num=mmd_kernel_num, mmd_kernel_mul=mmd_kernel_mul, margin_max=margin_max, margin_min=margin_min)
    if loss_type.lower() == "tune_loss":
        return tune_loss.Tune_Loss(ng=ng,nf=nf,nw=nw,margin=margin, alpha=alpha, beta=beta, p=p, r=r, mmd_kernel_num=mmd_kernel_num, mmd_kernel_mul=mmd_kernel_mul)
    if loss_type.lower() == "compact_triplet_mmd2":
        return compact_triplet_mmd2.Compact_Triplet_MMD2(ng=ng,nf=nf,nw=nw,margin=margin, alpha=alpha, beta=beta, p=p, r=r,q=q, mmd_kernel_num=mmd_kernel_num, mmd_kernel_mul=mmd_kernel_mul)
    
    raise NameError("Loss function not found")

def traceback(acc_cost_matrix : npt.DTypeLike):
    """ Encontra a path (matching) dado uma matriz de custo acumulado obtida a partir do cálculo do DTW.
    Args:
        acc_cost_matrix (npt.DTypeLike): matriz de custo acumulado obtida a partir do cálculo do DTW.

    Returns:
        Tuple[npt.ArrayLike, npt.ArrayLike]: coordenadas ponto a ponto referente a primeira e segunda sequência utilizada no cálculo do DTW.
    """
    rows, columns = np.shape(acc_cost_matrix)
    rows-=1
    columns-=1

    r = [rows]
    c = [columns]

    while rows != 0 or columns != 0:
        aux = {}
        if rows-1 >= 0: aux[acc_cost_matrix[rows -1][columns]] = (rows - 1, columns)
        if columns-1 >= 0: aux[acc_cost_matrix[rows][columns-1]] = (rows, columns - 1)
        if rows-1 >= 0 and columns-1 >= 0: aux[acc_cost_matrix[rows -1][columns-1]] = (rows -1, columns - 1)
        key = min(aux.keys())
    
        rows, columns = aux[key]

        r.insert(0, rows)
        c.insert(0, columns)
    
    return np.array(r), np.array(c)

def dump_hyperparameters(hyperparameters : Dict[str, Any], res_folder : str):
    with open(res_folder + os.sep + 'hyperparameters.json', 'w') as fw:
        json.dump(hyperparameters, fw)