##MPNN learning process of hydrogen atom(fig.1-4)

import H_xyz_MPNN_onetwoeight_fun
import matplotlib.pyplot as plt
import torch
from torch import nn, optim

data = H_xyz_MPNN_onetwoeight_fun.MetropolisHastings(16000)#The number of samples ranges from 2000 to 16000
# data = H_xyz_ones_fun.sample_x_a(10000)
print(data)
dataset = H_xyz_MPNN_onetwoeight_fun.MyDataset(data)
loader = H_xyz_MPNN_onetwoeight_fun.DataLoader(dataset, batch_size=16000, shuffle=True)#batch_size = number of samples

num_epochs = 2000
# loss = []
loss = torch.zeros(2000)
en = torch.zeros(2000)
r = torch.tensor([1.0, 0.0, 0.0])
r1 = torch.tensor([5.0, 5.0, 5.0])
# r = torch.tensor([data[:,0],0.0,0.0])
for epoch in range(num_epochs):
    # for i in range (10000):
    for n_batch, n_data in enumerate(loader):
        #  n_data = Variable(batch, requires_grad=True)
        # print('n',n_data.shape)
        H_xyz_MPNN_onetwoeight_fun.optimizer.zero_grad()
        psi_p, psi_g, Energy = H_xyz_MPNN_onetwoeight_fun.conservation_energy(n_data)
        # print("psi_g",psi_g)
        # print("psi_p",psi_p)
        potential_c = H_xyz_MPNN_onetwoeight_fun.potential(r)
        # error = ((psi_p-psi_g)**2).sum()
        # error = nn.MSELoss()(psi_g, psi_p) + ((potential_c+1) ** 2).sum() #1e-3 * H_xyz_ones_fun.potential(r1)
        # error = nn.MSELoss()(psi_g, psi_p) + 1e-2 *((H_xyz_ones_fun.potential(n_data)-potential_c-1) ** 2).sum()
        # error = (psi_p-psi_g).norm(p=2)+ 1e-2 *(torch.abs(H_xyz_ones_fun.potential(n_data)-potential_c-1) ** 2).sum()
        # error = (psi_p - psi_g).norm(p=2)#+H_xyz_ones_fun.potential(r1)
        # error = nn.MSELoss()(psi_g,psi_p) + 0.01*((torch.exp(-(1)*(psi_g)**2)*((H_xyz_ones_fun.potential(n_data)-potential_c-1) )** 2).sum())/n_data.numel()
        #+10*(potential_c-(-1)).norm(p=2)
        error = nn.MSELoss()(Energy*psi_g, psi_p)+0.001*(potential_c+1) ** 2
        # error = nn.MSELoss()(Energy * psi_g, psi_p) + 0.001*potential_c

        error.backward(retain_graph=True)
        H_xyz_MPNN_onetwoeight_fun.optimizer.step()
    loss[epoch] = error.item()
    en[epoch] = (H_xyz_MPNN_onetwoeight_fun.potential(r) - Energy).item()
    # print(loss)
    print('loss', error.item())
    print((H_xyz_MPNN_onetwoeight_fun.potential(r) - Energy).item())
    # print((H_xyz_ones_fun.potential(r) ).item())
torch.save(H_xyz_MPNN_onetwoeight_fun.potential.state_dict(),'ones_Potential_128_chu0.01_mc16000_6.pkl')
X = torch.arange(0,2000,1)
# plt.plot(X,loss)
# plt.plot(X,en)
torch.save(loss,'ones_loss_128_chu0.01_mc16000_6.pkl')
# torch.save(en,'ones_energy_128_chu0.01_rate0.1.pkl')
# plt.show()