    
import numpy as np
from numpy import typing as npt
from Losses import triplet_loss, triplet_mmd, compact_triplet_mmd, clustering_triplet_mmd, contrastive_loss, triplet_loss_offset, compact_triplet_mmd_offset, clustering_triplet_loss, hard_triplet_loss_avg, hard_triplet_loss_max, triplet_distillation, tune_loss, compact_triplet_mmd2, compact_triplet_mmd_scr, compact_triplet_mmd_slice, window_loss, window_loss2,compact_triplet_mmd_random, compact_triplet_mmdx, compact_triplet_mmd_gaussian, double_anchor_ctm, syn_compact_triplet_mmd
from typing import List, Tuple, Dict, Any
import torch
import json
import os

def define_loss(loss_type : str, ng : int, nf : int, nw : int, margin : torch.nn.Parameter, model_lambda : float, alpha : torch.nn.Parameter, beta : torch.nn.Parameter, p : torch.nn.Parameter, r : torch.nn.Parameter,q : torch.nn.Parameter, mmd_kernel_num : float, mmd_kernel_mul : float, margin_max : float, margin_min : float, nsl = int):
    if loss_type.lower() == "triplet_loss":
        return triplet_loss.Triplet_Loss(ng=ng, nf=nf, nw=nw, margin=margin, model_lambda=model_lambda)
    if loss_type.lower() == "triplet_mmd":
        return triplet_mmd.Triplet_MMD(ng=ng,nf=nf,nw=nw,margin=margin, model_lambda=model_lambda, alpha=alpha, p=p, r=r, mmd_kernel_num=mmd_kernel_num, mmd_kernel_mul=mmd_kernel_mul)
    if loss_type.lower() == "compact_triplet_mmd":
        return compact_triplet_mmd.Compact_Triplet_MMD(ng=ng,nf=nf,nw=nw,margin=margin, alpha=alpha, beta=beta, p=p, r=r, mmd_kernel_num=mmd_kernel_num, mmd_kernel_mul=mmd_kernel_mul)
    if loss_type.lower() == "syn_compact_triplet_mmd":
        return syn_compact_triplet_mmd.Syn_Compact_Triplet_MMD(ng=4, ns=4, nr=3, sng=2, sns=2,nw=nw,margin=margin, alpha=alpha, beta=beta, p=p, r=r, mmd_kernel_num=mmd_kernel_num, mmd_kernel_mul=mmd_kernel_mul)
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
        return compact_triplet_mmd2.Compact_Triplet_MMD2(ng=ng,nf=nf,nw=nw,margin=margin, alpha=alpha, beta=beta, p=p, r=r, mmd_kernel_num=mmd_kernel_num, mmd_kernel_mul=mmd_kernel_mul)
    if loss_type.lower() == "compact_triplet_mmd_scr":
        return compact_triplet_mmd_scr.Compact_Triplet_MMD_scr(ng=ng,nf=nf,nw=nw,margin=margin, alpha=alpha, beta=beta, p=p, r=r, mmd_kernel_num=mmd_kernel_num, mmd_kernel_mul=mmd_kernel_mul)
    if loss_type.lower() == "compact_triplet_mmd_slice":
        return compact_triplet_mmd_slice.Compact_Triplet_MMD_Slice(ng=ng,nf=nf,nw=nw,margin=margin, alpha=alpha, beta=beta, p=p, r=r, nsl=nsl, mmd_kernel_num=mmd_kernel_num, mmd_kernel_mul=mmd_kernel_mul)
    if loss_type.lower() == "window_loss":
        return window_loss.Window_Loss(ng=ng, nf=nf, nw=nw, margin=margin, model_lambda=model_lambda, r=r)
    if loss_type.lower() == "window_loss2":
        return window_loss2.Window_Loss2(ng=ng, nf=nf, nw=nw, margin=margin, model_lambda=model_lambda, r=r, p=p)
    if loss_type.lower() == "compact_triplet_mmd_random":
        return compact_triplet_mmd_random.Compact_Triplet_MMD_Random(ng=ng,nf=nf,nw=nw,margin=margin, alpha=alpha, beta=beta, p=p, r=r, mmd_kernel_num=mmd_kernel_num, mmd_kernel_mul=mmd_kernel_mul)
    if loss_type.lower() == "compact_triplet_mmdx":
        return compact_triplet_mmdx.Compact_Triplet_MMDX(ng=ng,nf=nf,nw=nw,margin=margin, alpha=alpha, beta=beta, p=p, r=r, mmd_kernel_num=mmd_kernel_num, mmd_kernel_mul=mmd_kernel_mul)
    if loss_type.lower() == "compact_triplet_mmd_gaussian":
        return compact_triplet_mmd_gaussian.Compact_Triplet_MMD_gaussian(ng=ng,nf=nf,nw=nw,margin=margin, alpha=alpha, beta=beta, p=p, r=r, mmd_kernel_num=mmd_kernel_num, mmd_kernel_mul=mmd_kernel_mul,mmin=margin_min, mmax=margin_max)
    if loss_type.lower() == "double_anchor_ctm":
        return double_anchor_ctm.Double_Anchor_CTM(ng=ng,nf=nf,nw=nw,margin=margin, alpha=alpha, beta=beta, p=p, r=r, mmd_kernel_num=mmd_kernel_num, mmd_kernel_mul=mmd_kernel_mul)
    raise NameError("Loss function not found")

def traceback(acc_cost_matrix : npt.DTypeLike):
    """ Encontra a path (matching) dado uma matriz de custo acumulado obtida a partir do cálculo do DTW.
    Args:
        acc_cost_matrix (npt.DTypeLike): matriz de custo acumulado obtida a partir do cálculo do DTW.

    Returns:
        Tuple[npt.ArrayLike, npt.ArrayLike]: coordenadas ponto a ponto referente a primeira e segunda sequência utilizada no cálculo do DTW.
    """
    rows, columns = np.shape(acc_cost_matrix)
    rows-=2
    columns-=2

    r = [rows]
    c = [columns]

    while rows != 0 or columns != 0:
        aux = {}
        if rows-1 >= 0: aux[acc_cost_matrix[rows -1][columns]] = (rows - 1, columns)
        if columns-1 >= 0: aux[acc_cost_matrix[rows][columns-1]] = (rows, columns - 1)
        if rows-1 >= 0 and columns-1 >= 0: aux[acc_cost_matrix[rows -1][columns-1]] = (rows -1, columns - 1)
        keys = list(aux.keys())
        key = min(keys)

        rows, columns = aux[key]

        r.insert(0, rows)
        c.insert(0, columns)
    
    return np.array(r), np.array(c)

def dump_hyperparameters(hyperparameters : Dict[str, Any], res_folder : str):
    with open(res_folder + os.sep + 'hyperparameters.json', 'w') as fw:
        json.dump(hyperparameters, fw)