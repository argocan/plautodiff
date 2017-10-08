import numpy as np
from pldiffer import Tensor
from pldiffer import Operations as op


x = Tensor(np.arange(0.0, 12.0*7, dtype=np.float64).reshape((12, 7)), diff=True)
y = Tensor(np.arange(0.0, 7.0*15, dtype=np.float64).reshape((7, 15)), diff=True)
bias = Tensor(np.zeros((1, 15), dtype=np.float64), diff=True)

print("\nX")
print(x)
print("\nY")
print(y)

z = op.matmul(x, y)
a1 = 0.5 * op.quadratic(z)
a2 = op.exp(bias)
a = a1 + a2
al = op.sigmoid(op.log(a))
f = op.sum(al)

print("\nFWD")
print(f)

f.calc_gradients()

print("\nG_X")
print(x.grad)

print("\nG_Y")
print(y.grad)

print("\nG_B")
print(bias.grad)



#####################VERIFY#####################################################
import torch
from torch.autograd import Variable

vx = Variable(torch.from_numpy(x.data), requires_grad=True)
vy = Variable(torch.from_numpy(y.data), requires_grad=True)
vb = Variable(torch.from_numpy(bias.data), requires_grad=True)

vz = torch.mm(vx, vy)
va1 = 0.5 * torch.pow(vz, 2)
va2 = torch.exp(vb)
va = va1 + va2
vl = torch.sigmoid(torch.log(va))
vf = torch.sum(vl)

vf.backward()

print("\nCHECK FWD")
ch = vf.data.numpy() - f.data
print(ch)

print("\nCHECK GRAD_X")
chx = np.sum(vx.grad.data.numpy() - x.grad)
print(chx)

print("\nCHECK GRAD_Y")
chy = np.sum(vy.grad.data.numpy() - y.grad)
print(chy)

print("\nCHECK_GRAD_B")
chb = np.sum(vb.grad.data.numpy() - bias.grad)
print(chb)

