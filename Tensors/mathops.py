import torch

torch.manual_seed(42)

x = torch.randn(2,4)
frand_vector = torch.randn(2,1)

print("----------Math Operations--------------")
#Types: Pointwise (one-to-one in size); Reduction; Comparison

activation_function = torch.tanh #"Activation Function" --> pointwise
#Easily replacable and implementable
print(activation_function(x))


sum = torch.sum(frand_vector)  # Sum of all elements
mean = torch.mean(frand_vector) 

print(f"{sum}\n{mean}") # Sum/Mean of all elements

#Matmul
A = torch.rand(2,2)
B = torch.rand(2,2)

print(f"{A}\n{B}\n")

Z = torch.matmul(A,B)
print(Z)


