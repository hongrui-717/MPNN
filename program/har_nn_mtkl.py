#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##MPNN learning process of harmonic oscillator(table1)

"""
Created on Mon Sep 28 17:19:56 2020

@author: rui
"""

from __future__ import division
import os, sys, time, random
import math
import scipy
from scipy import constants
import torch
from torch import nn, optim
from torch import autograd
from torch.autograd import grad
#import autograd.numpy as np
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.nn import functional as F
from scipy.constants import pi
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, title, show, xlabel, ylabel, legend

class Potential(nn.Module):
    def __init__(self):
        super(Potential,self).__init__()
        self.hidden0 = nn.Sequential(
            nn.Linear(1,32),
           # nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(32,32),
          #  nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(32,128),
          #  nn.BatchNorm1d(128),
            nn.ReLU()
         )
        self.hidden3 = nn.Sequential(
            nn.Linear(128,128),
          #  nn.BatchNorm1d(128),
            nn.ReLU()
         )
        self.out = nn.Sequential(
            nn.Linear(128,1),
            nn.Sigmoid()
            #nn.ReLU()
        )
        
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = x + self.hidden3(x)
        x = 12.5*self.out(x) #不理解
        #x = self.out(x)
        return x
    
def hermite(n,x):
    if n==0:
        return 1
    elif n==1:
        return 2*x
    else:
        return 2*x*hermite(n-1,x)-2*(n-1)*hermite(n-2,x) #recursion

def harmonic(m,h,w,n,x):
    #Normalization:
    norm=((m*w)/(math.pi*h))**(1/4)
    term1=(math.factorial(n))*(2**n)
    term2=(hermite(n,x)/math.sqrt(term1))
    expterms=(-1.0*m*w*x*x)/(2*h)
    #print(norm*term2,expterms,x)
    evalh=norm*term2*torch.exp(expterms)
    
    #print(norm,term1,term2,evalh)
    return evalh 
def init_wave_function(x): 
    #return harmonic(1,1,1,0,x)*harmonic(1,1,1,0,x)#This is the probability density, |Psi*Psi|**2 of the harmonic oscilator 
    return harmonic(1,1,1,2,x)

def harmonic_a(m,h,w,n,x):
    #Normalization:
    norm=((m*w)/(math.pi*h))**(1/4)
    term1=(math.factorial(n))*(2**n)
    term2=(hermite(n,x)/math.sqrt(term1))
    expterms=(-1.0*m*w*x*x)/(2*h)
    #print(norm*term2,expterms,x)
    evalh=norm*term2*np.exp(expterms)
    
    #print(norm,term1,term2,evalh)
    return evalh 
def init_wave_function_a(x): 
    #return harmonic(1,1,1,0,x)*harmonic(1,1,1,0,x)#This is the probability density, |Psi*Psi|**2 of the harmonic oscilator 
    return harmonic_a(1,1,1,2,x)

#def  ground_state_wave_function (x):
    
 #   psi_g = np.sqrt(1/(np.sqrt(np.pi)))*torch.exp(-(x**2)/2)
    #psi_g =  torch.exp(-(x*6+0.1)**2) * ((x-0.2) ** 2) + torch.exp(-(x*6-0.4)**2) * (x ** 2)
    #psi_g = torch.exp(-(x*5)**2)
   # psi_g = psi_g / (torch.norm(psi_g)*np.sqrt(0.02))
   # print('gy',torch.norm(psi_g)*np.sqrt(0.06))
#    return psi_g
#def  ground_state_wave_function_a (x):
    
 #   psi_g = np.sqrt(1/(np.sqrt(np.pi)))*np.exp(-(x**2)/2)
    #psi_g = psi_g / np.linalg.norm(psi_g)
    #print(psi_g.shape)
 #   return psi_g

potential = Potential()
optimizer = torch.optim.Adam(potential.parameters(), lr = 0.001)

def new_prob_dist(batch):
    output = init_wave_function(batch)
    output.requires_grad_(True)
    potential_energy = potential(batch)
   # potential_energy.requires_grad_(True)
    first_der = grad(output, batch, grad_outputs = torch.ones_like(batch), 
                    create_graph=True, retain_graph=True, 
                   only_inputs=True,
                   allow_unused=True
                  )[0] 
   
    kinetic_energy = grad(first_der, batch, grad_outputs = torch.ones_like(batch), 
                    create_graph=True, retain_graph=True, 
                   only_inputs=True,
                   allow_unused=True
                  )[0]
    psi = (-(kinetic_energy/2) + (potential_energy*output))
   # print('psi',output.shape)
   # E = torch.norm(psi)*np.sqrt(0.06)
    #E = torch.sqrt(torch.mm(psi.T,psi))/(torch.sqrt(torch.mm(output.T,output)))
    #print(((psi/output).sum()))
    E = ((psi/output).sum())/output.numel()
    # E = (psi * output).sum() * 0.002
    # E = 2.5
    # psi=psi/E
    #print('E=',E)
    #print('E=',E/(torch.norm(output)*np.sqrt(0.06)))
    print('E=',E)
    #print('E=',E/torch.sqrt(torch.mm(output.T,output)))
    #psi=output-psi
    #print(conserve_energy)
    return psi,E,output


#x_range = [-5,5]

def probability(x):
    pro=init_wave_function_a(x)**2
    return pro
    
#Better to get from data
x_range = [-5,5]
def sample_x(size):
   x = (x_range[0] - x_range[1]) * torch.rand(size,1) + x_range[1]
   x1=torch.linspace(-1,1,size)
   x = x1.reshape(size,1)
   print(x.shape)
   return x
def MetropolisHastings(n):
   x=0.5

   cc=np.zeros(n)
   for i in range(n):  
       #x1 = random.uniform(-1,1)  
       dx=random.uniform(-0.6,0.6)
       x1=x+dx

       #-5 < x1 < 5 and 
       if probability(x)<probability(x1):
           x=x1
   
       elif np.random.uniform(0,1)<(probability(x1)/probability(x)):
           x=x1
       cc[i]=x
   cc1 = cc.reshape(n,1)
   cc2 = torch.from_numpy(cc1)
  # print(cc)
   return cc2


class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data.float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.data)
    
data = MetropolisHastings(2500)
# data = sample_x(5000)
       
dataset = MyDataset(data)
loader = DataLoader(dataset, batch_size = 32, shuffle = True)
loss = torch.zeros(1000)
num_epochs = 1000
# loss = []
for epoch in range(num_epochs):
    for n_batch, batch in enumerate(loader):
        n_data = Variable(batch, requires_grad=True)
        optimizer.zero_grad()
        psi_p, Energy,psi_g = new_prob_dist(n_data)
        # total_energy = init_wave_function (n_data)-new_prob_dist(n_data)
        r = torch.tensor([1.0])
        potential_c = potential(r)
        error = nn.MSELoss()(Energy * psi_g, psi_p)+(potential_c-0.5)**2
        # error = (total_energy**2).sum()
      
        error.backward(retain_graph=True)
        
        optimizer.step()
      #  print(potential(data))
      #  print(error)
    # loss.append(error)
    print('loss',error.item())
    loss[epoch] = error.item()
x_test = MetropolisHastings(2500)
tensor_1 = torch.FloatTensor(5)
x_test=x_test.type_as(tensor_1)
x_test = Variable(x_test, requires_grad=True)
# En= torch.norm(new_prob_dist(x_test))
acc_v = ((potential(x_test).detach().numpy() - (0.5*(x_test.detach().numpy() **2) ) )**2).sum()
print('acc_v=',acc_v)#每次不一样
print(Energy)
plt.scatter(data.detach().numpy(),.5* pow(data.detach().numpy(),2))
plt.scatter(x_test.detach().numpy(), potential(x_test).detach().numpy())
title("potential")
legend(['Exact','Learned'])
show()

# plt.scatter(x_test.detach().numpy(), init_wave_function (x_test).detach().numpy())
# plt.scatter(x_test.detach().numpy(), new_prob_dist(x_test).detach().numpy())
# legend(['Exact','Learned'])
# show()
torch.save(potential.state_dict(),'har_m_Potential_3c.pkl')
X = torch.arange(0,1000,1)
torch.save(loss,'har_loss_m_3c.pkl')