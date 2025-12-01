import torch

device = "mps"

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

x = torch.tensor([2.0], requires_grad=True, device=device)
y = x * 3 + 1
y.backward()

print("device:", x.device)
print("x:", x.item())
print("y:", y.item())
print("dy/dx:", x.grad.item())
