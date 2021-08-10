##MPNN fun hydrogen atom

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
            nn.ReLU()
            # nn.Tanh()
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(128, 128),
            # nn.Dropout(p=0.5),
            # nn.BatchNorm1d(128),
            nn.ReLU()
            # nn.Tanh()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(128, 128),
            #  nn.BatchNorm1d(128),
            nn.ReLU()
            # nn.Tanh()
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU()
            # nn.Tanh()
        )
        #self.hidden4 = nn.Sequential(
         #   nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
        #    nn.ReLU()
        #)
        #self.hidden5 = nn.Sequential(
        #    nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
        #    nn.ReLU()
        #)
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

#torch.manual_seed(0)
potential = Potential()
# params = list(potential.named_parameters())
# # params[0] = torch.randn(x.shape, device=x.device, dtype=x.dtype) * 0.0
for x in potential.parameters():
    x.data = torch.randn(x.shape, device=x.device, dtype=x.dtype) * 0.01
# params[9] = torch.randn(x.shape, device=x.device, dtype=x.dtype) * 0.0-(0.01*torch.ones(128,128))
optimizer = torch.optim.Adam(potential.parameters(), lr=.001)

# print('p',params[7])
# print('p',params[8])
# print('p',params[9])
#
# X = torch.arange(-5.05, 5, 0.1)
# #Y = torch.arange(-5.05, 5, 0.1)
# Z = torch.arange(-5.05, 5, 0.1)
# # Y = torch.zeros(20)
# #Z = torch.zeros(10201)
# Y = torch.zeros(10201)
# C = np.meshgrid(X, Z)
# listtuple1 = []
# for i in range(len(C)):
#     listtuple1 += list(C[i])
# tensorlisttuple1 = torch.tensor(listtuple1)
# C1 = tensorlisttuple1.reshape(2, -1).transpose(1, 0)
# C1[:, [1, 0]] = C1[:, [0, 1]]
# C2 = np.insert(C1, 1, values=Y, axis=1)
# # C1 = X1,Y1.reshape(-1,2)
# x = C2.type(torch.float32)
# #print('x', x)
# x = Variable(x, requires_grad=True)
#
# p = potential(x)
# x_coord = x[:, 0]
# y_coord = x[:, 1]
# z_coord = x[:, 2]
#
# p1=p
# #Z1=Z1.reshape(101,101)
# p2=p1.reshape(101,101)
#
# fig = plt.figure()
# #ax3 = plt.axes(projection='3d')
# ax3 = Axes3D(fig)
#
# X1,Z1 = np.meshgrid(X, Z)
# R1 = p2
#
#
#
#
# ax3.plot_surface(X1,Z1,R1.detach().numpy(),rstride = 1,cstride = 1,cmap='YlGnBu')

# plt.show()
def probability(x, y, z):
    pro = psi_radial_1s_a(x, y, z) ** 2
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
    output = psi_radial_1s(x_coord, y_coord, z_coord)
    output.requires_grad_(True)
    potential_energy = potential(batch).squeeze()
    potential_energy =potential_energy.reshape(-1, 1)
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
    # print(potential_energy.shape)

    psi = (-(kinetic_energy / 2) + (potential_energy * output))
    # psi = psi.reshape(-1,1)

    # print(psi.shape)
    # E=torch.norm(psi)*np.sqrt(0.06)

    # E=torch.sqrt(torch.mm(psi.T,psi))/(torch.sqrt(torch.mm(output.T,output)))
    E = ((psi / output).sum()) / output.numel()
    # E = (psi*output).sum()*0.001
    # print('E=',E)
    # E = -0.5
    psi1 = psi / E
    # psi=output-psi

    return psi, output, E


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

        # if 0.5<x1<10 and 0.5<y1<10 and 0.5<z1<10 and probability(x,y,z)<probability(x1,y1,z1):

        # if x1**2+y1**2+z1**2 > 0.5 :x1 ** 2 + y1 ** 2 + z1 ** 2 > 0.01 and
        if probability(x, y, z) < probability(x1, y1, z1):
            cc[num] = x1
            dd[num] = y1
            ff[num] = z1
            num += 1
            x = x1
            y = y1
            z = z1

            # elif 0.5<x1<10 and 0.5<y1<10 and 0.5<z1<10 and np.random.uniform(0,1)<(probability(x1,y1,z1)/probability(x,y,z)):
        elif  np.random.uniform(0, 1) < probability(x1, y1, z1) / probability(x,y,z):
            # print('a',(probability(x1)/probability(x)))
            cc[num] = x1
            dd[num] = y1
            ff[num] = z1
            num += 1
            x = x1
            y = y1
            z = z1
        # else:
        #    cc[num] = x
        #   dd[num] = y
        #  ff[num] = z
        # num+=1

        # print(x)
        # cc1=copy.copy(cc)
        #  cc[i]=x
        # dd[i]=y
        # ff[i]=z

        # num+=1
        # gg[i]=-1/(np.sqrt(x**2+y**2+z**2))
    ee = torch.cat((cc[:num, :], dd[:num, :], ff[:num, :]), dim=1)
    #  print('ee',ee.shape)

    # ee1 = ee.reshape(n,3)
    # ee1 = torch.from_numpy(ee)

    # zz+=y
    # print(ee1.shape)
    # print(zz)
    return ee
x_range=[-5,5]
y_range=[-5,5]
z_range=[-5,5]
def sample_x_a(size):
 #   '''
  #  Uniform random x sampling within range
   # '''
    x = (x_range[0] - x_range[1]) * torch.rand(size,1) + x_range[1]
    y = (y_range[0] - y_range[1]) * torch.rand(size, 1) + y_range[1]
    z = (z_range[0] - z_range[1]) * torch.rand(size, 1) + z_range[1]
    #x1=torch.linspace(-2,2,size)
    # x1=torch.linspace(-5,5,size)
    # y1=torch.linspace(-5,5,size)
    # z1=torch.linspace(-5,5,size)
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