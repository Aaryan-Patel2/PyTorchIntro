import torch


print("---------------------Init, Indexing/Slicing----------------------")

rand_vector = torch.tensor([[1,2,3,4,5,6,7,8,9,10]]) #Uniform distribution for rand; Gaussian distribution for randn
torch.manual_seed(42) #Random seed --> Maintain same values after compilation
x = torch.randn(2,4)

print(x[0][2]) #Indexing
x[0][2] = 32.2 #In-place replacement; NOTE --> Raises error if the tensor involved has auto_grad on
#Reason: This leads to overwriting values --> improper gradient calculations (consistency ruined)

print(f"In place mutation:{x}")

y = x[0:1, 0:2] #slicing (2*4 matrix) --> (1*2) vector (row vector)

#Indexing                      
y_rows = x[0, :]
y_columns = x[: ,0]
                                                                            
#P.S --> Using the :/, operators --> first row/column being kept while eliminating everything else
y_T = y.T #Transpose operation --> traditional column vector

print(f"Column of x: {y}")
print(f"Transpose of a vector split from x: {y_T}")

print(f"Columns:{y_columns}\nRows:{y_rows}")

flattened_rand = rand_vector.flatten() #Dimensions are that of a 1 * 10 tensor (matrix) due to double brackets; should be a vector of a single bracket
indices = torch.tensor([0, 2, 4, 6, 8])# Correct indexing to select elements from the second dimension of a 2D tensor
selected_elements = rand_vector[0, indices]  #NOTE: Use 0 to specify the first row, then use indices for columns

print(f"{selected_elements}\n")

rand_zeroes = torch.zeros(2,3) #Works same way with ones & empties
print(rand_zeroes)

#The rand-vector from earlier can be generalized to be an a-range operation
rand_vector2 = torch.arange(1,11,1)
print(rand_vector2)

print(rand_vector.device)


arand = torch.linspace(0, 1, 5) #Creates a vector that has a constant level of magnitude change as we expand dimensions
print(arand)

#Identity Matrix --> Useful for residual/skip connections + direct initialization; Generates in nxn size (n is the input)

i3 = torch.eye(3)

print(i3)