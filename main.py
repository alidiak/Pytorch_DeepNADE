#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 12:46:27 2021

@author: Alexander Lidiak
"""

import torch
import torch.nn as nn
from Deep_NADE import DeepNADE
from utils import rand_ordering
from torchvision import datasets, transforms
#import sys

def train(model, train_loader, optimizer, device): 
    model.train() 
    N_iter = len(train_loader.dataset)/train_loader.batch_size
    for batch_ind,  (target_data, _) in enumerate(train_loader):
        target_data = target_data.view(target_data.shape[0],-1).to(device)
        optimizer.zero_grad()
        _, _, loss = model(N_samples=train_loader.batch_size,\
                                 x_train=target_data)
        # The forward computes cost and performs backprop with x_train specified
        optimizer.step()
        if batch_ind % (N_iter/20) == 0: # update every 5% completed
            print('Training ', str(round((batch_ind/N_iter)*100)), \
                  '% complete.', ' Current loss: ', str(loss.item()), '\n')

def test(model, test_loader, device):
    model.eval()
    cost_total = 0.0
    N_iter = len(train_loader.dataset)/train_loader.batch_size
    for batch_ind, (test_data, _) in enumerate(test_loader):
        test_data = test_data.view(test_data.shape[0],-1).to(device)
        _, _, cost = model(N_samples=train_loader.batch_size,\
                                 x_train=test_data)
        cost_total+=cost.item()/N_iter
        
    print('Cost Total: ', str(cost_total))
    return cost_total

concat_mask = False
lr = 1
num_epochs = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

# This code is unnecessary in Pytorch update, but as of March 5, 2020 was necessary
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
## 

# Borrowed from MNIST examples
transform=transforms.Compose([transforms.ToTensor()])

training_set = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_set = datasets.MNIST('../data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(training_set, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

# input layer size (reshaping MNIST to be 1D)
L=len(training_set[0][0].view(-1))

# hidden layer size
H = L

#TODO: Add random ordering

# specify the structure/model of the DeepNADE Neural Network
# can be anything so long as it ends in a sigmoid (so that 0<p(xi)<1)
if concat_mask: L_in = 2*L
else: L_in=L
FFNN = nn.Sequential(nn.Linear(L_in,H), nn.Sigmoid(), 
        nn.Linear(H,L), nn.Sigmoid()) # sigmoid  

model = DeepNADE(FFNN, device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs/10) #gamma=0.9)

for epoch in range(num_epochs):
    train(model, train_loader, optimizer, device)
    test(model, test_loader,device)
    scheduler.step()


torch.cuda.empty_cache()

