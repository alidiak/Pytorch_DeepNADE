#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 28 2021

@author: alex
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from Deep_NADE import DeepNADE
from utils import rand_ordering


concat_mask = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# input layer size
L=100

# hidden layer size
H = 2*L

# specify the structure/model of the DeepNADE Neural Network
# can be anything so long as it ends in a sigmoid (so that 0<p(xi)<1)
if concat_mask: L_in = 2*L
else: L_in=L
model = nn.Sequential(nn.Linear(L_in,H), nn.Sigmoid(), 
        nn.Linear(H,L), nn.Sigmoid()) # sigmoid  

# test sample generation
sample_size = 1000
rand_s = torch.tensor(torch.randint(0,2,(sample_size,L)), dtype=torch.float)
test = DeepNADE(model, rand_s)

# test on random sample set
prob, _ = test(x=rand_s.to(device))

# Now test inputing a random ordered list (for order-agnostic training)
rand_order_list = rand_ordering(sample_size,L)
prob, samples = test(N_samples=sample_size, order=rand_order_list)

pars=list(model.parameters())

torch.cuda.empty_cache()


        


