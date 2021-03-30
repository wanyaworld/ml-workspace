import torch

lr = 1e-5
epochs = 2000

a = torch.tensor((3.0), requires_grad = True)

# learnable parameters (weights)
w1 = torch.tensor((1.0), requires_grad = True)
w2 = torch.tensor((1.0), requires_grad = True)
w3 = torch.tensor((1.0), requires_grad = True)
w4 = torch.tensor((1.0), requires_grad = True)

for t in range(epochs):
  # intermediate layers
  b = w1*a*a + 2
  c = w2*a
  d = w3*b - w4*c*c*c

  # loss function
  L = (10 - d) * (10 - d)

  print("epoch(" + str(t) + ") " + "loss: " + str(L.item()))

  L.backward()

  # We manually update grad, so disable auto grad.
  with torch.no_grad():
    w1 -= lr * w1.grad;
    w2 -= lr * w2.grad;
    w3 -= lr * w3.grad;
    w4 -= lr * w4.grad;

  # Since pytorch accumulates grads, zero them.
    w1.grad = None
    w2.grad = None
    w3.grad = None
    w4.grad = None
