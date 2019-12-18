# THis is to test if 3 grad calculations are equal to 2 grad cal

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Hyper Parameters 
input_size = 10
hidden_size = 50
num_classes = 10
num_epochs = 5
learning_rate = 0.001


# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
model = Net(input_size, hidden_size, num_classes)
model.cuda()   
    
# Loss and Optimizer
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

for epoch in range(num_epochs):
    with torch.enable_grad():
         optimizer.zero_grad()
         output = model(torch.rand(10,10).cuda())
         loss = criterion(output, torch.rand(10,10).cuda())
         ones = [torch.ones_like(p) for p in model.parameters()]
         dl_dw = torch.autograd.grad(loss, model.parameters(), create_graph=True)
         d2l_dw2 = torch.autograd.grad(dl_dw, model.parameters(), ones, create_graph=True)
         sum_dl2_dw2 = sum([g.sum() for g in d2l_dw2])
         print(sum_dl2_dw2)
         loss.backward()
         optimizer.step()

'''
torch.manual_seed(0)
torch.cuda.manual_seed(0)
model = Net(input_size, hidden_size, num_classes)
model.cuda()   
    
# Loss and Optimizer
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

for epoch in range(num_epochs):
    with torch.enable_grad():
         optimizer.zero_grad()
         output = model(torch.rand(10,10).cuda())
         loss = criterion(output, torch.rand(10,10).cuda())
         ones = [torch.ones_like(p) for p in model.parameters()]
         dl_dw = torch.autograd.grad(loss, model.parameters(), create_graph=True)
         d2l_dw2 = torch.autograd.grad(dl_dw, model.parameters(), ones, create_graph=True)
         sum_dl2_dw2 = sum([g.sum() for g in d2l_dw2])
         print(sum_dl2_dw2)
'''
