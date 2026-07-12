"""
00. Pytorch Fundamentals
Resource notebook : https://www.learnpytorch.io/00_pytorch_fundamentals/
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
1. Introduction to Tensors 
pytorch tensors are created using torch.tensor() function.  LINK : https://docs.pytorch.org/docs/2.13/tensors.html

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
print(tensor.shape) # shape of the tensor


### Random Tensors
# Random tensors are tensors of arbitrary size filled with random numbers.
random_tensor = torch.rand(3, 4)
print(random_tensor)
print(random_tensor.ndim) # 2-d tensor

#Creating a random tensor with similar shape to an image tensor (height, width, color channels)
#Troch random tensors - https://pytorch.org/docs/stable/generated/torch.rand.html
random_image_size_tensor = torch.rand(224, 224, 3) # height, width, color channels
print(random_image_size_tensor.shape, random_image_size_tensor.ndim) # shape and dimension of the tensor


#Tensor of zeros or ones
#Torch zeros tensor - https://pytorch.org/docs/stable/generated/torch.zeros.html
ones_tensor = torch.ones(3, 4)
print(ones_tensor)
print(ones_tensor.dtype)


###Creatinfg a range of tensors and tensors-like
#tensor.arange() - https://pytorch.org/docs/stable/generated/torch.arange.html  to make a range of tensors
Arranged_tensor = torch.arange(start=1, end=11, step=1) # create a tensor from 1 to 10
print(Arranged_tensor) # 1-d tensor from 1 to 10

#Creating a tensor-like of another tensor
tensor_zeros_like = torch.zeros_like(Arranged_tensor) # create a tensor of zeros with the same shape as random_tensor
print(tensor_zeros_like) 

##Tensor Datatypes
#Tensors can have different datatypes. The default datatype is float32.
#recuires_grad=False means that the tensor will not be tracked for gradients during backpropagation.
float_32_tensor = torch.tensor([3.0, 6.0, 9.0], dtype=torch.float32, device=None, requires_grad=False)
print(float_32_tensor.dtype) 
"""

""" The three main issues that i'll probably run into when working with pytorch & deep learning are:
1. Tensor datatypes
2. Tensor shapes
3. Tensor device (CPU vs GPU)
"""

"""
# Manipulating tensors (tensor operations)
Tensor operations include :
    addition   subtraction   multiplication(element-wise)   division(element-wise)   matrix multiplication   reshaping   indexing/slicing   aggregation (sum, mean, etc.)   broadcasting
"""
"""
tensor = torch.tensor([1,2,3,4,5])

#ADDING TENSOR
print(tensor+10) 

#SUBTRACTING TENSOR
print(tensor-10)

#MULTIPLYING TENSOR
print(tensor*10)

#DIVIDING TENSOR
print(tensor/10)
"""

"""
When performing matrix multiplication---

the inner dimensions must be the same.:
(3,2) @ (3,2) will not work
(3,2) @ (2,3) will work
(2,3) @ (3,2) will work

The output shape will be the outer dimensions of the two matrices being multiplied.
(3,2) @ (2,3) = (3,3)

To perform matrix multiplication in PyTorch, you can use the @ operator or the torch.matmul() function.
"""
Tensor_A = torch.tensor([[1, 2], 
                         [3, 4], 
                         [5, 6]]) # shape (3,2)

Tensor_B = torch.tensor([[7, 8], 
                         [9, 10], 
                         [11, 12]]) # shape (3,2)


"""
Whem we multiply Tensor_A and Tensor_B, we get a runtime error; To fix this, we can transpose one of the tensors to make the inner dimensions match.
    A transpose switches the axes of a inputed tensor.
    To transpose a tensor in PyTorch, you can use the .T attribute or the torch.transpose() function.

Tensor_B_T = Tensor_B.T 
tensor_matmul_AB = torch.matmul(Tensor_A, Tensor_B_T)
print(tensor_matmul_AB, '\n', tensor_matmul_AB.shape)
"""



