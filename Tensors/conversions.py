import torch
print("----------Conversion-----------------------")

rand_vector = torch.rand(1,10)
frand_vector = rand_vector.float() #Works inversely with integers
print(frand_vector)

rand_vector = rand_vector.add_(1) #In-Place
print(f"{rand_vector}\n")

#Works similarly for device conversion as a method --> tensor.to('gpu/cpu')
