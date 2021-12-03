import torch

a = torch.rand(2,3,4,dtype=torch.float32)
b = torch.rand(2,3,4,dtype=torch.float32)

c = a*b
d = c.mean(0)

print(a)
print(b)
print(c)
print(d)
print(d.shape)