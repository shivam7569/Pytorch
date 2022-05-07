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
batch, m, n, p = 10, 5, 6, 2
t1 = torch.rand((batch, m, n))
t2 = torch.rand((batch, n, p))
t3 = torch.bmm(t1, t2)
print(t3)
