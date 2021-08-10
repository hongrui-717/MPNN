##QPNN learning process of hydrogen atom(fig.2,3)

from __future__ import division
import os, sys, time, random
import math
import scipy
from scipy import constants
import torch
from torch import nn, optim
from torch import autograd
from torch.autograd import grad
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.nn import functional as F
from scipy.constants import pi
from mpl_toolkits.mplot3d import Axes3D

class Potential(nn.Module):
    def __init__(self):
        super(Potential, self).__init__()
        self.symme_mirror = False
        self.hidden0 = nn.Sequential(
            nn.Linear(3, 128),
            # nn.Dropout(p=0.5),
            # nn.BatchNorm1d(128),
            # nn.ReLU()
            nn.Tanh()
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(128, 128),
            # nn.Dropout(p=0.5),
            # nn.BatchNorm1d(128),
            # nn.ReLU()
            nn.Tanh()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(128, 128),
            #  nn.BatchNorm1d(128),
            # nn.ReLU()
            nn.Tanh()
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            # nn.ReLU()
            nn.Tanh()
        )
        # self.hidden4 = nn.Sequential(
        #     nn.Linear(128, 128),
        #     # nn.BatchNorm1d(128),
        #     nn.ReLU()
        # )
        # self.hidden5 = nn.Sequential(
        #     nn.Linear(128, 128),
        #     # nn.BatchNorm1d(128),
        #     nn.ReLU()
        # )
        self.out = nn.Sequential(
            nn.Linear(128, 1)
            # nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.reshape(-1,3)
        #   x = torch.cat((x,1/x.norm(dim=1,keepdim=True)),dim=1)
        x1 = -x
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = x + self.hidden3(x)
        # x = x + self.hidden4(x)
        # x = x + self.hidden5(x)
        x = self.out(x)
        if self.symme_mirror:
            # x1 = -x
            x1 = self.hidden0(x1)
            x1 = self.hidden1(x1)
            x1 = self.hidden2(x1)
            x1 = x1 + self.hidden3(x1)
            # x1 = x1 + self.hidden4(x1)
            # x = x + self.hidden5(x)
            x1 = self.out(x1)

            return (x+x1)/2
        else:
            return x



def psi_radial_1s(x, y, z):
    a = 1 / (np.sqrt(pi))
    # a=2
    return a * torch.exp(-(torch.sqrt(x ** 2 + y ** 2 + z ** 2)))


def psi_radial_1s_a(x, y, z):
    a = 1 / (np.sqrt(pi))
    # a=2
    return a * np.exp(-(np.sqrt(x ** 2 + y ** 2 + z ** 2)))

# def psi_radial_2p(x, y, z):
#     a = (np.sqrt(3)) / (np.sqrt(4 * pi))
#     b = (torch.sqrt(x ** 2 + y ** 2 + z ** 2)) * torch.exp(-(torch.sqrt(x ** 2 + y ** 2 + z ** 2)) / 2)
#     c = z / ((torch.sqrt(x ** 2 + y ** 2 + z ** 2)))
#     return a * b * c
#
#
# def psi_radial_2p_a(x, y, z):
#     a = (np.sqrt(3)) / (np.sqrt(4 * pi))
#     b = (np.sqrt(x ** 2 + y ** 2 + z ** 2)) * np.exp(-(np.sqrt(x ** 2 + y ** 2 + z ** 2)) / 2)
#     c = z / ((np.sqrt(x ** 2 + y ** 2 + z ** 2)))
#     return a * b * c
# torch.manual_seed(0)
potential = Potential()
params = list(potential.named_parameters())
print(params)
# for x in potential.parameters():
#     x.data = torch.randn(x.shape, device=x.device, dtype=x.dtype) * 0.0
# params[9] = torch.randn(x.shape, device=x.device, dtype=x.dtype) * 0.0+0.01
optimizer = torch.optim.Adam(potential.parameters(), lr=.001)

def probability(x, y, z):
    pro = psi_radial_1s_a(x, y, z) ** 2
    # pro = psi_radial_2p_a(x, y, z) ** 2
    return pro


def conservation_energy(batch):
    batch.requires_grad_(True)
    x_coord = batch[:, 0]
    x_coord = x_coord.reshape(-1, 1)
    # print(x_coord.shape)
    x_coord.requires_grad_(True)
    y_coord = batch[:, 1]
    y_coord = y_coord.reshape(-1, 1)
    y_coord.requires_grad_(True)
    z_coord = batch[:, 2]
    z_coord = z_coord.reshape(-1, 1)
    z_coord.requires_grad_(True)
    # output = torch.abs(psi_radial_1s(x_coord, y_coord, z_coord))
    output = psi_radial_1s(x_coord, y_coord, z_coord)
    output.requires_grad_(True)
    potential_energy = potential(batch).squeeze()
    potential_energy = potential_energy.reshape(-1, 1)
    # print('exp',torch.exp(-100* (output ** 2)))
    #  print(potential_energy.shape)
    potential_energy.requires_grad_(True)
    # potential_energy = .5*(x_coord**2 + y_coord**2).squeeze()
    # print(potential_energy)
    dHdx = grad(output, x_coord, grad_outputs=torch.ones_like(x_coord),
                create_graph=True, retain_graph=True,
                only_inputs=True,
                allow_unused=True
                )[0]
    d2Hdx2 = grad(dHdx, x_coord, grad_outputs=torch.ones_like(x_coord),
                  create_graph=True, retain_graph=True,
                  only_inputs=True,
                  allow_unused=True
                  )[0]
    dHdy = grad(output, y_coord, grad_outputs=torch.ones_like(y_coord),
                create_graph=True, retain_graph=True,
                only_inputs=True,
                allow_unused=True
                )[0]
    d2Hdy2 = grad(dHdy, y_coord, grad_outputs=torch.ones_like(y_coord),
                  create_graph=True, retain_graph=True,
                  only_inputs=True,
                  allow_unused=True
                  )[0]
    dHdz = grad(output, z_coord, grad_outputs=torch.ones_like(z_coord),
                create_graph=True, retain_graph=True,
                only_inputs=True,
                allow_unused=True
                )[0]
    d2Hdz2 = grad(dHdz, z_coord, grad_outputs=torch.ones_like(y_coord),
                  create_graph=True, retain_graph=True,
                  only_inputs=True,
                  allow_unused=True
                  )[0]
    kinetic_energy = d2Hdx2 + d2Hdy2 + d2Hdz2
    conserve_energy = kinetic_energy/(2*output) - potential_energy


    return conserve_energy


h = .01


def taylor_approx_x(batch):
    batch.requires_grad_(True)
    x_coord = batch[:, 0]
    x_coord.requires_grad_(True)
    x_coord1 = x_coord + h
    x_coord2 = x_coord - h
    x1_coord1 = torch.unsqueeze(x_coord1, 1)
    x2_coord2 = torch.unsqueeze(x_coord2, 1)

    y_coord = batch[:, 1]
    y_coord.requires_grad_(True)
    y_coord = torch.unsqueeze(y_coord, 1)
    z_coord = batch[:, 2]
    z_coord.requires_grad_(True)
    z_coord = torch.unsqueeze(z_coord, 1)
    batch_forward = torch.cat([x1_coord1, y_coord, z_coord], 1)
    batch_back = torch.cat([x2_coord2, y_coord, z_coord], 1)

    partial_x = (conservation_energy(batch_forward) - conservation_energy(batch_back)) / (2 * h)
    return partial_x


def taylor_approx_y(batch):
    batch.requires_grad_(True)
    x_coord = batch[:, 0]
    x_coord.requires_grad_(True)
    x_coord = torch.unsqueeze(x_coord, 1)
    z_coord = batch[:, 2]
    z_coord.requires_grad_(True)
    z_coord = torch.unsqueeze(z_coord, 1)

    # x1_coord = torch.unsqueeze(x1_coord,1)
    y_coord = batch[:, 1]
    y_coord.requires_grad_(True)
    y1 = y_coord + h
    y2 = y_coord - h
    y1_coord = torch.unsqueeze(y1, 1)
    y2_coord = torch.unsqueeze(y2, 1)
    batch_forward = torch.cat([x_coord, y1_coord, z_coord], 1)
    batch_back = torch.cat([x_coord, y2_coord, z_coord], 1)

    partial_y = (conservation_energy(batch_forward) - conservation_energy(batch_back)) / (2 * h)
    return partial_y

def taylor_approx_z(batch):
    batch.requires_grad_(True)
    x_coord = batch[:, 0]
    x_coord.requires_grad_(True)
    x_coord = torch.unsqueeze(x_coord, 1)
    y_coord = batch[:, 1]
    y_coord.requires_grad_(True)
    y_coord = torch.unsqueeze(y_coord, 1)

    # x1_coord = torch.unsqueeze(x1_coord,1)
    z_coord = batch[:, 2]
    z_coord.requires_grad_(True)
    z1 = z_coord + h
    z2 = z_coord - h
    z1_coord = torch.unsqueeze(z1, 1)
    z2_coord = torch.unsqueeze(z2, 1)
    batch_forward = torch.cat([x_coord,y_coord, z1_coord], 1)
    batch_back = torch.cat([x_coord,y_coord, z2_coord], 1)

    partial_z = (conservation_energy(batch_forward) - conservation_energy(batch_back)) / (2 * h)
    return partial_z

def MetropolisHastings(n):
    x = 0.5
    y = 0.5
    z = 0.5
    cc = torch.zeros((n * 2, 1))
    dd = torch.zeros((n * 2, 1))
    ff = torch.zeros((n * 2, 1))
    num = 0
    num1 = 0
    # gg=np.zeros(n)
    # for i in range(n):
    while num < n and num1 < n * 2:
        num1 += 1

        x1 = x + random.uniform(-0.2, 0.2)
        # dy=random.uniform(-0.2,0.2)
        y1 = y + random.uniform(-0.2, 0.2)
        # dz=random.uniform(-0.2,0.2)
        z1 = z + random.uniform(-0.2, 0.2)
#x1**2+y1**2+z1**2 > 0.5 and
        if probability(x, y, z) < probability(x1, y1, z1):
            cc[num] = x1
            dd[num] = y1
            ff[num] = z1
            num += 1
            x = x1
            y = y1
            z = z1

            # elif 0.5<x1<10 and 0.5<y1<10 and 0.5<z1<10 and np.random.uniform(0,1)<(probability(x1,y1,z1)/probability(x,y,z)):
        elif np.random.uniform(0, 1) < probability(x1, y1, z1) / probability(x,y,z):
            # print('a',(probability(x1)/probability(x)))
            cc[num] = x1
            dd[num] = y1
            ff[num] = z1
            num += 1
            x = x1
            y = y1
            z = z1

    ee = torch.cat((cc[:num, :], dd[:num, :], ff[:num, :]), dim=1)

    return ee
x_range = [-5,5]
y_range = [-5,5]
z_range = [-5,5]
def sample_x_a(size):
    x = (x_range[0] - x_range[1]) * torch.rand(size, 1) + x_range[1]
    y = (y_range[0] - y_range[1]) * torch.rand(size, 1) + y_range[1]
    z = (z_range[0] - z_range[1]) * torch.rand(size, 1) + z_range[1]
    # print('range',x_range[1])
    # x1=torch.linspace(0,1,size)
    # y1=torch.linspace(0,1,size)
    # z1=torch.linspace(0,1,size)
    # x = x1.reshape(size,1)
    # y = y1.reshape(size,1)
    # z = z1.reshape(size,1)
    r = torch.cat((x,y,z),dim=1)
   # print(r)
    return r

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

# data = MetropolisHastings(2500)
data = sample_x_a(16000)
print(data)
dataset = MyDataset(data)
loader = DataLoader(dataset, batch_size=16000, shuffle=True)

num_epochs = 2000
loss = []
loss = torch.zeros(2000)
x = torch.tensor([1.0,0,0])
for epoch in range(num_epochs):
    for n_batch, batch in enumerate(loader):
        # n_data = Variable(batch, requires_grad=True)




        error = (taylor_approx_x(batch)**2 + taylor_approx_y(batch)**2+ taylor_approx_z(batch)**2).mean()+(potential(x)+1)**2
#
        optimizer.zero_grad()
        error.backward(retain_graph=True)
        optimizer.step()
    # loss.append(error)
    print('loss11', error.item())
    loss[epoch] = error.item()
torch.save(loss,'one_loss128_yelu_tanh_n16000_8.pkl')
torch.save(potential.state_dict(),'one_Potential128_yelu_tanh_n16000_8.pkl')