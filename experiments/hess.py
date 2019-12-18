# this program is to confirm the correctness of the hessian code vs hessian vector mul code

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,2)
        self.fc2 = nn.Linear(2,1)
    def forward(self, x):
        x = self.fc2(F.relu(self.fc1(x)))
        return x
net = Net().cuda()


input = Variable(torch.randn(1,1,1)).cuda()


net.zero_grad()

out = net(input)

import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.4)
criterion = nn.MSELoss()

data = [(1,3), (2,6), (3,9), (4,12), (5,15), (6,18)]

for epoch in range(100):
    for i, data2 in enumerate(data):
        X, Y = iter(data2)
        X, Y = Variable(torch.FloatTensor([X]), requires_grad=True).cuda(), Variable(torch.FloatTensor([Y]), requires_grad=False).cuda()
        optimizer.zero_grad()
        y_pred = net(X)
        output = criterion(y_pred, Y)
        output.backward()
        optimizer.step()
    if (epoch % 20 == 0.0):
        print("Epoch {} - loss: {}".format(epoch, output))

i = Variable(torch.randn(1)).cuda()
b = 2*i*i   
