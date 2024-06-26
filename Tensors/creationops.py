import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------Init, Indexing/Slicing-----------------------------------------------

rand_vector = torch.tensor([[1,2,3,4,5,6,7,8,9,10]])
torch.manual_seed(42) #Random seed --> Maintain same values after compilation
x = torch.randn(2,4)

print(x[0][2]) #Indexing
x[0][2] = 32.2 #In-place replacement; NOTE --> Raises error if the tensor involved has auto_grad on
#Reason: This leads to overwriting values --> improper gradient calculations (consistency ruined)

print(x)

y = x[0:1, 0:2] #slicing (2*4 matrix) --> (1*2) vector (row vector)

#P.S --> Using the ":" operator --> all rows/columns being maintained 
y_T = y.T #Transpose operation --> traditional column vector

print(y)
print(y_T)

flattened_rand = rand_vector.flatten() #Dimensions are that of a 1 * 10 tensor (matrix) due to double brackets; should be a vector of a single bracket
indices = torch.tensor([0, 2, 4, 6, 8])# Correct indexing to select elements from the second dimension of a 2D tensor
selected_elements = rand_vector[0, indices]  #NOTE: Use 0 to specify the first row, then use indices for columns

print(selected_elements)

# -------------------------------------Conversion-----------------------------------------------
frand_vector = rand_vector.float() #Works inversely with integers
print(frand_vector)

rand_vector = rand_vector.add_(1) #In-Place
print(rand_vector)

# -------------------------------------Math Operations-----------------------------------------------
#Types: Pointwise (one-to-one in size); Reduction; Comparison

activation_function = torch.tanh #"Activation Function" --> pointwise
#Easily replacable and implementable
print(activation_function(x))


sum = torch.sum(frand_vector)  # Sum of all elements
mean = torch.mean(frand_vector) 

print(sum, mean) # Sum/Mean of all elements




