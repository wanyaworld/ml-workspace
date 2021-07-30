import torch
import torch.nn as nn
import torch.optim as optim

tensor = torch.randn(3, 4, requires_grad=True)
parameter = (torch.nn.Parameter(tensor),)
sol = torch.nn.Parameter(torch.arange(12, dtype=torch.float).view(3, 4))
loss_fn = nn.MSELoss()
optimizer = optim.SGD(parameter, lr=0.01)

for i in range(3000):
    optimizer.zero_grad()
    prediction = parameter[0]
    loss = loss_fn(prediction, sol)
    loss.backward()
    optimizer.step()
print (parameter[0])
print ('\n\n vs \n\n')
print (sol)
