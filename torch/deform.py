import elasticdeform.torch as etorch
import numpy as np
import torch

displacement_val = np.random.randn(2, 3, 3) * 5
X_val = np.random.rand(200, 300)
dY_val = np.random.rand(200, 300)

# construct PyTorch input and top gradient
displacement = torch.tensor(displacement_val)
X = torch.tensor(X_val, requires_grad=True)
dY = torch.tensor(dY_val)

# the deform_grid function is similar to the plain Python equivalent,
# but it accepts and returns PyTorch Tensors
X_deformed = etorch.deform_grid(X, displacement, order=3)

# the gradient w.r.t. X can be computed in the normal PyTorch manner
X_deformed.backward(dY)
print(X.grad)

# x_deformed = elasticdeform.deform_random_grid(data,sigma=)
def deform(y):
    np.random.randn()
    displacement_val = np.random.randn(y.shape) * 5
    displacement = torch.tensor(displacement_val)
    y_deformed = etorch.deform_grid(y, displacement, order=3)
    return y_deformed
