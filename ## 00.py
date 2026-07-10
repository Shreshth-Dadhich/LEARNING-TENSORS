# 00. Pytorch Fundamentals
# Resource notebook : https://www.learnpytorch.io/00_pytorch_fundamentals/

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Introduction to Tensors 
# pytorch tensors are created using torch.tensor() function.  LINK : https://docs.pytorch.org/docs/2.13/tensors.html

#scalar:
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim) # 0-d tensor
print(scalar.item()) # get the python integer from a tensor

#vector:
vector = torch.tensor([7, 7])
print(vector)
print(vector.ndim) # 1-d tensor
print(vector.shape) # shape of the tensor

#matrix:
matrix = torch.tensor([[7, 8], [9, 10]])
print(matrix)
print(matrix.ndim) # 2-d tensor
print(matrix.shape) # shape of the tensor
print(matrix[0]) # get the first row of the matrix

#tensor:
tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]])
print(tensor)
print(tensor.ndim) # 3-d tensor
