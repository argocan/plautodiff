import numpy as np
from pldiffer import *
from pldiffer import Operations as op
import torch
from torch.autograd import Variable

a = np.ones((2,10,10))
b = np.ones((2,10)).reshape(2,10)[:,:,np.newaxis]

print(np.einsum('ijk,ijl->ijl',a,b).squeeze(axis=2).shape)


a = np.array([0,1,2])
b = np.array([0,1,2])

#OUTER
c = np.einsum('i,j -> ij', a, b)
d = np.outer(a, b)

#DIAGONAL
c = np.diag(a)
#d = np.einsum('ii', a[:,np.newaxis])
print(c)
#print(d)

def print_f(x):
    print(x.data.numpy())

print("****************************************************")
print("****************************************************")
print("****************************************************")
a = np.array([[1.0,1.0,.5]],dtype=np.float32)
ad = Tensor(a)
at = Variable(torch.from_numpy(a), requires_grad=True)
sd = op.softmax(ad)
st = torch.nn.functional.softmax(at)
st.register_hook(print_f)
md = op.log(op.sum(sd))
mt = torch.log(torch.sum(st))
md.calc_gradients()
mt.backward()
print("***")
print(sd)
print(ad.grad)
print(at.grad.data.numpy())

