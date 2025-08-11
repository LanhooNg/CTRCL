import torch

a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([1, 3, 9, 4])
c = torch.where(a < b, torch.tensor(1), torch.tensor(0))
print(c)