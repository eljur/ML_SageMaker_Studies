import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Complete this classifier
class SimpleNet(nn.Module):
    
    ## TODO: Define the init function
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(SimpleNet, self).__init__()
        
        # define all layers, here
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.prelu = nn.PReLU(1)
        self.fc3 = nn.Linear(hidden_dim,output_dim)
        self.out_act = nn.Sigmoid()
        
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        x = F.relu(self.fc1(x))
        x = self.dout(x)
        x = self.fc2(x)
        x = self.prelu(x)
        x = self.fc3(x)
        x = self.out_act(x)
        return x
    
