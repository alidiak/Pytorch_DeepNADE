#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 29 2021

@author: alex
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np

# useful function for getting random orderings
def rand_ordering(K, L):
     
    rand_order_list = np.tile(np.arange(L),(K,1))
    rand_mat = np.random.rand(rand_order_list.shape[0], rand_order_list.shape[1])
    rand_ind = np.argsort(rand_mat)
    rand_order_list=rand_order_list[ np.arange(L), rand_ind]

    return rand_order_list

