import numpy as np
import torch

device = "cpu"  # if torch.cuda.is_available() else "cpu"

# ========== TENSOR INITIALISATION ========== #

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True)
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.shape)
print(my_tensor.device)

x = torch.empty(size=(3, 3))
print(x)
x = torch.ones(size=(3, 3))
print(x)
x = torch.zeros(size=(3, 3))
print(x)
x = torch.rand(size=(3, 3))
print(x)
x = torch.eye(3, 3)
print(x)
x = torch.arange(start=0, end=5, step=1)
print(x)
x = torch.linspace(start=0.1, end=1, steps=10)
print(x)
x = torch.empty(size=(3, 3)).normal_(mean=0, std=1)
print(x)
x = torch.empty(size=(3, 3)).uniform_(0)
print(x)
x = torch.diag(torch.ones(3))
print(x)

# ========== TENSOR TYPE CASTING ========== #

x = torch.arange(4)
print(x)
print(x.bool())
print(x.short())
print(x.long())
print(x.double())
print(x.float())
print(x.half())

# ========== TENSOR TO/FROM NUMPY ========== #

np_array = np.ones((3, 3))
x = torch.from_numpy(np_array)
print(x)
np_array = x.numpy()
print(np_array)

# ========== TENSOR OPERATIONS ========== #

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# Addition
z = torch.add(x, y)
print(z)
z = x + y
print(z)

# Subtraction
z = torch.subtract(x, y)
print(z)
z = x - y
print(z)

# Division
z = torch.true_divide(x, y)
print(z)
z = x / y
print(z)

# Exponentiation
z = x.pow(2)
print(z)
z = x ** 2
print(z)

# InPlace addition
print(z.add_(x))
z = x < 2
print(z)

# Matrix Multiplication
a = torch.rand((2, 5))
b = torch.rand((5, 3))
c = torch.mm(a, b)
print(c)
c = a.mm(b)
print(c)

# Matrix Exponentiation
d = torch.rand((3, 3))
e = d.matrix_power(3)
print(e)

# ElementWise Multiplication
z = x * y
print(z)

# Dot Product
z = torch.dot(x, y)
print(z)

# Batch Matrix Multiplication
batch, m, n, p = 2, 5, 6, 2
t1 = torch.rand((batch, m, n))
t2 = torch.rand((batch, n, p))
t3 = torch.bmm(t1, t2)
print(t3)

# Broadcasting Example
t1 = torch.rand((3, 3))
t2 = torch.rand((1, 3))
z = t1 ** t2
print(z)

# Other Useful Operations
t = torch.tensor([[-1, 3, 2], [4, 5, 6], [7, 8, 9]])
z = torch.sum(t, dim=0)
print(z)
values, indices = torch.max(t, dim=1)
print(values, indices)
values, indices = torch.min(t, dim=1)
print(values, indices)
z = torch.abs(t)
print(z)
z = torch.argmax(t, dim=1)
print(z)
z = torch.mean(t.float(), dim=0)
print(z)
z = torch.eq(x, y)
print(z)
values, indices = torch.sort(t, dim=1, descending=False)
print(values, indices)
z = torch.clamp(t, min=2, max=5)
print(z)
z = t.ndimension()
print(z)
z = t.numel()
print(z)

# ========== TENSOR INDEXING ========== #

batch_size = 10
features = 25
x = torch.rand((batch_size, features))
z = x[0]
print(z.shape)
z = x[:, 0]
print(z.shape)
z = x[2, :10]
print(z.shape)

indices = [2, 5, 8]
z = x[indices]
print(z)

rows = torch.tensor([2, 3, 5])
cols = torch.tensor([1, 4, 6])
z = x[rows, cols]
print(z.shape)

z = x[(x < 2) | (x > 8)]
print(z.shape)

z = x[x.remainder(2) == 0]
print(z.shape)

z = torch.where(x > 0.3, x, 0)
print(z)
