import torch
from torch import nn
import math

from model_arch import SemiSupervisedConsistencyModelTorch


def kl_divergence(y_true, y_pred):
    # if there are no labeled training examples:
    if list(y_true.size)[0] == 0:
        # There are no labeled training examples
        return 0
    else:
        return torch.nn.KLDivLoss()(y_true, y_pred)
        # return tf.keras.backend.mean(tf.keras.losses.kl_divergence(y_true, y_pred))

def supervised_loss(self, data):
    return kl_divergence(y_true, y_pred)

def consistency_loss(self, data):
    return 0.0

def custom_loss(self, data):
    """
    Compute total loss:
        supervised + unsupervised consistency loss.

    Parameters
    ----------
    data : tuple
        The output of the generator.

    Returns
    -------
    loss_value : scalar
        Total loss.
    loss_sup : scalar
        Supervised loss.
    loss_usup : scalar
        Unsupervised loss.
    pair_sup : a tuple of tensors
        Ground truth labels and predictions on labeled examples.
    pair_usup : a tuple of tensors
        Predictions on two differently transformed labeled and unlabeled examples.
    """

    x, y, labeled = data

    # number of unique labeled and labeled+unlabeled images
    n_labeled = torch.count_nonzero(labeled) // 2
    n = list(x.size)[0] // 2

    # compute predictions on all examples
    pred = self(x)

    # n_labeled = tf.cast(tf.math.count_nonzero(labeled), tf.int32) // 2
    # n = tf.shape(x)[0] // 2

    # compute predictions on all examples
    pred = self(x)

    # separate labeled images from the rest
    yl = tf.concat((y[:n_labeled, ...], y[:n_labeled, ...]), axis=0)
    predl = tf.concat((pred[:n_labeled, ...], pred[n:(n + n_labeled), ...]), axis=0)

    # separate differently transformed
    pred1, pred2 = pred[:n, ...], pred[n:, ...]

    # supervised loss
    loss_sup = self.loss(yl, predl)

    # unsupervised loss made symmetric (e.g. KL divergence is not symmetric)
    loss_usup = (self.loss(pred1, pred2) + self.loss(pred2, pred1)) / 2

    # total loss: supervised + weight * unsupervised consistency
    loss_value = loss_sup + self.p.alpha * loss_usup

    return loss_value, loss_sup, loss_usup, (yl, predl), (pred1, pred2)

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

model = SemiSupervisedConsistencyModelTorch()

criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam()
epochs = 1000

for t in range(0, epochs):
    # forward pass
    y_pred = model(x)

    # compute loss
    loss = criterion(y_pred, y)
    if t % 10 == 9:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
