#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.true_divide(w_avg[k], len(w))
    return w_avg


def soft(ratios):
    weights_array = np.array(ratios)
    soft_v = []
    for r in ratios:
        s = np.exp(r) / np.exp(weights_array).sum()
        soft_v.append(s)

    return soft_v


def InverseNorm(weights_list):
    weights_array = np.array(weights_list)
    inv_norm_r = 1 - weights_array / weights_array.sum()
    return inv_norm_r.tolist()


def Norm(weights_list):
    weights_array = np.array(weights_list)
    inv_norm_r = weights_array / weights_array.sum()
    return inv_norm_r.tolist()


def FedWeightedAvg(w, cur_weights_dict, use_soft=False):
    weights = []
    for client in cur_weights_dict.keys():
        cur_client_weights_list = []
        for k, v in zip(cur_weights_dict[client].keys(), cur_weights_dict[client].values()):
            cur_client_weights_list.append(v)
        weights.append(1 / sum(cur_client_weights_list))
    weights = Norm(weights)
    if use_soft:
        weights = soft(weights)
    # 保证权重和网络选定数目相同
    assert len(weights) == len(w)
    # weights = torch.as_tensor(weights, dtype=torch.float32, device='cuda:0')
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        w_avg_k = 0
        for weight, w_a in zip(weights, w):
            w_avg_k += weight * w_a[k]
        w_avg[k] = w_avg_k
    return w_avg
