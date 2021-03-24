import torch
from torch import nn


def kl_divergence(y_true, y_pred):
    # if there are no labeled training examples:
    if y_true.shape[0] == 0:
        # There are no labeled training examples
        return 0
    else:
        return torch.nn.KLDivLoss()(y_true, y_pred)
        # return tf.keras.backend.mean(tf.keras.losses.kl_divergence(y_true, y_pred))

# def supervised_loss(yl, predl):
#     return kl_divergence(yl, predl)
#
# def consistency_loss(y1, y2):
#     return 0.0

def custom_loss(data, pred, p):
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
    n = x.shape[0] // 2

    # separate labeled images from the rest
    yl = torch.cat((y[:n_labeled, ...], y[:n_labeled, ...]), dim=0)
    predl = torch.cat((pred[:n_labeled, ...], pred[n:(n + n_labeled), ...]), dim=0)

    # separate differently transformed
    pred1, pred2 = pred[:n, ...], pred[n:, ...]

    # supervised loss
    loss_sup = kl_divergence(yl, predl)

    # unsupervised loss made symmetric (e.g. KL divergence is not symmetric)
    loss_usup = (kl_divergence(pred1, pred2) + kl_divergence(pred2, pred1)) / 2

    # total loss: supervised + weight * unsupervised consistency
    loss_value = loss_sup + p.alpha * loss_usup

    return loss_value, loss_sup, loss_usup, (yl, predl), (pred1, pred2)


