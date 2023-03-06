from periodic_activations import SineActivation, CosineActivation
from Data import ToyDataset
from torch import nn
import torch

class Model(nn.Module):
    def __init__(self, activation, hiddem_dim):
        super(Model, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(400, hiddem_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(400, hiddem_dim)
        
        self.fc1 = nn.Linear(hiddem_dim, 2)
    
    def forward(self, x):
        #x = x.unsqueeze(1)
        # print('1=',x)
        x = self.l1(x)
        # print('2=', x)
        # x = torch.nn.functional.tanh(x)
        # print('3=',x)
        # print('t=',type(x))
        x = self.fc1(x)
        # print('4=', x)
        # x = torch.nn.functional.tanh(x)
        # print('5=', x)
        return x
