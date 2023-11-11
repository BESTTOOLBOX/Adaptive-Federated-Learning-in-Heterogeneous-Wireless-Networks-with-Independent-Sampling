#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from torch import nn
#from quantize import quantize, compress_tensor,decompress_tensor
# from models.compute_dp_sgd_privacy_lib import apply_dp_sgd_analysis

def FedAvg(w_old, w):
    w_avg = copy.deepcopy(w[0])
    w_avg_q = copy.deepcopy(w[0])


    for k in w_avg.keys():
        w_avg[k] = w_avg[k] - w_avg[k]
        w_avg_q[k] = w_avg_q[k] - w_avg_q[k]

        for i in range(0, len(w)):
            w_avg[k] = w[i][k] + w_avg[k]#

        # w_avg[k] = torch.true_divide(w_avg[k], len(w)).cpu()
        w_avg[k] = w_avg[k]/len(w)
        w_avg_q[k] = w_old[k] + w_avg[k]#


    return w_avg_q



'''
    num_bits = num_bits1
    w_avg = copy.deepcopy(w[0])
    w_avg_q = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]

        w_avg[k] = torch.true_divide(w_avg[k], len(w)).cpu()
        #(np.array(w_avg[k]).size)
        if np.array(w_avg[k]).size != 1:
            w_avg_q[k] = quantize(torch.Tensor(w_avg[k]), num_bits=num_bits)
        else:
            w_avg_q[k] = w_avg[k]
'''


