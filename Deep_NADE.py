#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Jan 28 2021

@author: Alexander Lidiak
This model takes as input a FFNN, and a target/training data set. It converts 
the FFNN into a Deep NADE model. Forward options include N_samples which will 
generate that number of samples, x which will run the forward given an input
data set x, and order, which should be a list of orderings (0-L), which the
autoregressive property of the DeepNADE will follow. Use rand_ordering function
in Utils to generate random orders and an order-agnostic DeepNADE that should
increase the model's accuracy and make a greater set of conditional 
probabilities tractable. Also, a mask concatenation can be applied if the model
input is twice that of the output - this can also improve performance.

For more information on DeepNADEs and the inspiration for this lib see:
https://arxiv.org/abs/1605.02226, Uria, B., Côté, M. A., Gregor, K., Murray, I.
, & Larochelle, H. (2016). Neural autoregressive distribution estimation. 
The Journal of Machine Learning Research, 17(1), 7184-7220.

Thanks also to Caleb Sanders for help with GPU testing/capability

"""

import numpy as np
import torch
import torch.nn as nn

class DeepNADE(nn.Module): # takes a FFNN model as input
            
    def __init__(self, model, device=None): 
        super(DeepNADE, self).__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else: self.device = device
        self.model = model.to(self.device)
        # input layer size (may be x2 if using concat mask)
#        self.x_train=x_train.to(self.device)
        self.M = self.model[0].in_features 
        self.D = self.model[-2].out_features
        self.mask_concat=False
        
        if self.M/self.D==2: self.mask_concat=True
            
    def forward(self, N_samples=None, x=None, order=None, x_train=None):
        
        self.to(self.device)
        J_x = 0 # running cost function
        
        # TODO: modify this as x_train can also specify number of samples needed
        if N_samples is None and x is None: 
            raise ValueError('Must enter samples or the number of samples to' \
                             ' be generated')
            
        if N_samples is None and x is not None: 
            N_samples, sample = x.shape[0], False
            x.to(self.device)

        if N_samples is not None and x is None: 
            sample = True 
            x = torch.zeros([N_samples,self.D],dtype=torch.float).to(self.device)
            
        if x_train is not None: 
            x_train=x_train.to(self.device)
        
#        PROB=torch.ones([N_samples])
        PROB=torch.zeros([N_samples]).to(self.device)
        
        if order is None: # autoregressive ordering = sequential - [x0,x1... xL]
            order = np.tile(np.arange(self.D),(N_samples,1))
        
        for d in range(self.D):
                
            # masks enforce the autoregressive property
            if d==0: mask=torch.zeros_like(x) 
            else: mask[range(N_samples),order[:, d-1:d].squeeze()]=1 
            # masking enforces the autoregressive property with a given order
            
            # dictates order for next pass and next masking
            od_1 = order[:,d:d+1].squeeze() 
                                    
            # run the model and get the probabilities for xd
            if self.mask_concat: # concatenates mask to input if desired
                out=self.model(torch.cat((mask*x,mask),dim=1))
            else: 
                out=self.model(mask*x)
                
            if d==0 and not torch.all(out>=0): # only doing for d=0 to save time
                raise ValueError('Input model requires positive and definite outputs'\
                ' in final layer. A Sigmoid activation function is recommended.')
                
            vi=out[range(N_samples),od_1].squeeze()
            
            # Sampling probability is determined by the separate conditionals
            if sample:
                
                # sampling routine:
                m = torch.distributions.Bernoulli(vi)
                x[range(N_samples),od_1]=m.sample()
            
            # Accumulate PPSI based on which state (s) was sampled
            # Recommended in NADE paper (but has scaling issues even with 
            # Sigmoid normalization) adding instead keeps the value tractable
#            PROB=PROB*(torch.pow(vi,x[range(N_samples),od_1]) + \
#                       torch.pow((1-vi),(1-x[range(N_samples),od_1])))
            PROB+=(vi*x[range(N_samples),od_1] + \
                 (1-vi)*(1-x[range(N_samples),od_1]))
                
            # Accumulate and backpropagate NLL here as the mask/ordering matters. 
            if x_train is not None:
                sample_ind = torch.randint(low=0,high=self.D,size=(1,)) #uniform sample from dist
                x_target = x_train[:,sample_ind].squeeze()
                J_xd = (self.D/(self.D-d+1))*(x_target*torch.log(vi+1e-6)+\
                    (1-x_target)*torch.log(1-vi+1e-6))
                # Have to add a small offset to log functions so Nan is not 
                # backpropagated if/when vi is 0.
                
                # mean will average over the different samples and/or orders
                J_xd.mean().backward()
                # grad is accumulated for each d and order here    
                
                J_x += J_xd.detach().mean()/self.D
            
        return PROB/self.D, x, J_x

            