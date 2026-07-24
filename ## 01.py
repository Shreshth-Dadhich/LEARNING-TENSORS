from torch import nn
import torch
import matplotlib.pyplot as plt
import pandas as pd


#Create known parameters
weight = 0.7
bias = 0.3

#Create
start = 0
end = 1
step = 0.02
X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight*X + bias 

print(X[:10],y[:10])