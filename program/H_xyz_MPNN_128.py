##result of MPNN
##test hydrogen atom energy and potential(fig.1-4)

from __future__ import division
import torch
from torch import nn, optim
from torch.autograd import grad
import numpy as np
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt
from scipy.constants import pi
from mpl_toolkits.mplot3d import Axes3D
import H_xyz_ones_fun

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


potential = Potential()

# potential.load_state_dict(torch.load('ones_Potential.pkl'))
potential.load_state_dict(torch.load('ones_Potential_128_chu0.01_mc16000_9.pkl'))#Call learning results
# losstest = torch.load('ones_loss_128_chu0.01_mc10000_7.pkl')
# print('testl',losstest)
# potential.load_state_dict(torch.load('ones_Potential_chu0.01_lam0.001.pkl'))#good
# potential.load_state_dict(torch.load('ones_32Potential_chu0.01_lam0.001.pkl'))
# potential.load_state_dict(torch.load('twop_Potential_yelu_tanh_n2000.pkl'))

def psi_radial_1s(x, y, z):
    a = 1 / (np.sqrt(pi))
    # a=2output
    return a * torch.exp(-(torch.sqrt(x ** 2 + y ** 2 + z ** 2)))

def psi_radial_1s_a(x, y, z):
    a = 1 / (np.sqrt(pi))
    # a=2
    return a * np.exp(-(np.sqrt(x ** 2 + y ** 2 + z ** 2)))

# def fun_v(x,y,z):
#     a = 1 / (np.sqrt(pi))
#     if a * np.exp(-(np.sqrt(x ** 2 + y ** 2 + z ** 2))) > 1e-3:
#        return 1.0
#     else:
#         return a * np.exp(-(np.sqrt(x ** 2 + y ** 2 + z ** 2)))

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
    #  print(potential_energy.shape)torch.exp(- (1) * (output ** 2))
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
    # E = (psi * output).sum() * 0.001
    # print('E=',E)
    # E = -0.5
    psi1 = psi / E
    # psi=output-psi

    return psi, output, E

def sample_x_a(size):
 #   '''
  #  Uniform random x sampling within range
   # '''
    #x = (x_range[0] - x_range[1]) * torch.rand(size,1) + x_range[1]
    #x1=torch.linspace(-2,2,size)
    x1=torch.linspace(-5,5,size)
    y1=torch.linspace(-5,5,size)
    z1=torch.linspace(-5,5,size)
    x = x1.reshape(size,1)
    y = y1.reshape(size,1)
    z = z1.reshape(size,1)
    r = torch.cat((x,y,z),dim=1)
   # print(r)
    return r
x_range = [-5,5]
y_range = [-5,5]
z_range = [-5,5]
def sample_x(size):
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
def sample_x_b(size):
    X = torch.linspace(-1,1,size)
    Y = torch.linspace(-1,1,size)
    Z = torch.linspace(-1,1,size)

    C = np.meshgrid(X, Y, Z)
    listtuple1 = []
    for i in range(len(C)):
        listtuple1 += list(C[i])
    tensorlisttuple1 = torch.tensor(listtuple1)
    C1 = tensorlisttuple1.reshape(3, -1).transpose(1, 0)
    x = C1.type(torch.float32)
   # print(r)
    return x

x_test = H_xyz_ones_fun.MetropolisHastings(10000)
# x_test = sample_x_a(5000)
print(x_test)
psi_pt,psi_gt,Energyt = conservation_energy(x_test)
print('EM',Energyt)
# error_a = nn.MSELoss()(Energyt*psi_gt, psi_pt)
# print('loss',error_a)
# x = sample_x(2000)
xb = sample_x_b(20)#20*20*20=8000samples
# print('bn',xb)
# X = torch.arange(-5.05, 5, 0.4)
#Y = torch.arange(-5.05, 5, 0.1)
# Y = torch.arange(-5.05, 5, 0.4)
# # Y = torch.zeros(20)
# #Z = torch.zeros(10201)
# Z = torch.arange(-5.05, 5, 0.4)
# C = np.meshgrid(X, Y, Z)
# listtuple1 = []
# for i in range(len(C)):
#     listtuple1 += list(C[i])
# tensorlisttuple1 = torch.tensor(listtuple1)
# C1 = tensorlisttuple1.reshape(3, -1).transpose(1, 0)
# x = C1.type(torch.float32)
# x = Variable(x, requires_grad=True)
# print(potential(x).reshape(1,2))
# print((1/(x[:,0]**2+x[:,1]**2+x[:,2]**2)))
print(potential(xb).shape)
print('poten',(torch.abs(potential(xb).reshape(1,8000)+torch.sqrt((1/(xb[:,0]**2+xb[:,1]**2+xb[:,2]**2))))).sum()/8000)#error of potential
# C1[:, [1, 0]] = C1[:, [0, 1]]
# C2 = np.insert(C1, 1, values=Z, axis=1)
# # C1 = X1,Y1.reshape(-1,2)
# x = C1.type(torch.float32)
# print(x.shape)
X = torch.arange(-5.05, 5, 0.1)
#Y = torch.arange(-5.05, 5, 0.1)
Y = torch.arange(-5.05, 5, 0.1)
# Y = torch.zeros(20)
#Z = torch.zeros(10201)
Z = torch.zeros(10201)
C = np.meshgrid(X, Y)
listtuple1 = []
for i in range(len(C)):
    listtuple1 += list(C[i])
tensorlisttuple1 = torch.tensor(listtuple1)
C1 = tensorlisttuple1.reshape(2, -1).transpose(1, 0)
C1[:, [1, 0]] = C1[:, [0, 1]]
C2 = np.insert(C1, 1, values=Z, axis=1)
# C1 = X1,Y1.reshape(-1,2)
x = C2.type(torch.float32)
#print('x', x)
x = Variable(x, requires_grad=True)
print(x)
psi_p1, psi_g1, Energy1 = conservation_energy(x)
#print(Energy1)
#print('psi', psi_p1)

p = potential(x)
eta = (torch.abs(p).sum())/x.numel()
print('eta',eta)
# p2=p.reshape(50,50,50)
# print('p_test',p2)
# p1 = conservation_energy(x)
# Better to get from data
x_coord = x[:, 0]
y_coord = x[:, 1]
z_coord = x[:, 2]
# p_v = p*fun_v(x_coord,y_coord,z_coord).detach().numpy()
# print('y',y_coord)
potential_c = potential(torch.tensor([[1.0,0,0]]))
print('p',potential_c)

#Z1=init_wave_function(x_coord,y_coord,z_coord)

psi_p1 = psi_p1.reshape(101,101)

p1=p-potential_c - 1
E1 = Energy1-potential_c - 1
print('e',Energy1)
#Z1=Z1.reshape(101,101)
p2=p.reshape(101,101)
ppp1 = p2 + 1/torch.sqrt(x_coord**2 + z_coord**2).reshape(101,101)
print('V',p2.shape)

# dx = 0.1
# dy = 0.1
# grad = torch.arange(-5, 5, 10 / 100)
# grad = grad.reshape(1, 100)
# for i in range(101):
#     grad_x1 = ((p2[i, 1:] - p2[i, :-1]) / (dx))
#     grad_y1 = ((p2[1:, i] - p2[:-1, i]) / (dy))
#     grad_2 = grad_x1 + grad_y1
#     grad_2 = grad_2.reshape(1, 100)
#     # grad_3 = torch.from_numpy(grad_2)
#     grad = torch.cat((grad, grad_2), 0)
# print('grad',grad[2:,:].shape)
        # grad = np.append(grad,grad_2,axis=0)


fig = plt.figure()
#ax3 = plt.axes(projection='3d')
ax3 = Axes3D(fig)

X1,Y1 = np.meshgrid(X, Y)
print(X1.shape)
R1 = p2
R2 = psi_radial_1s(x_coord, y_coord, z_coord).reshape(101, 101)
R3 = psi_p1.reshape(101,101)
R4 = -1/torch.sqrt(x_coord**2 + z_coord**2).reshape(101,101)
R5 = R2-R3
R6 = R2**2
# R = H_xyz_ones_fun.MetropolisHastings(10201).reshape(101,101)



# plt.figure(num=1, figsize=(12, 10), dpi=100)
ax3.plot_surface(X1,Y1,R1.detach().numpy(),rstride = 1,cstride = 1,cmap='YlGnBu')#rainbow，YlGnBu
# # ax3.scatter(X1,Y1,R2.detach().numpy())#rainbow，YlGnBu
ax3.set_ylabel('y')
ax3.set_xlabel('x')
ax3.set_zlabel('V')
# # ax3.grid(False)
# # ax3.set_xticks([])
# # ax3.set_yticks([])
# # ax3.set_zticks([])
# # plt.axis('off')
# # plt.subplots_adjust(0, 0, 1, 1)
ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)
plt.show()
