import torch
from omegaconf import OmegaConf
from torch import nn
import deform
import pyoneer_main.func as func
import os
import pyoneer_main.improc as improc


def kl_divergence(y_true, y_pred):
    # if there are no labeled training examples:
    if y_true.shape[0] == 0:
        # There are no labeled training examples
        return 0
    else:
        return torch.mean(torch.nn.KLDivLoss(reduction='batchmean')(y_pred.log(), y_true))
        # return tf.keras.backend.mean(tf.keras.losses.kl_divergence(y_true, y_pred))

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

    inputs, y, labeled = data

    x = inputs[0]
    transform_parameters = inputs[1:]

    # number of unique labeled and labeled+unlabeled images
    n_labeled = torch.count_nonzero(labeled)
    n = x.shape[0]

    # -- transformation --
    # get a transform function
    transform = getattr(func, 'get_batch_transform_' + p.transform.apply_func) \
        (*transform_parameters, **p.transform.params_apply)

    t_x = transform(x)  # transform input images
    x = torch.cat((x, t_x), dim=0)  # form a batch to feed to the network

    # if network outputs and labels also need to be transformed (as in the segmentation case):
    if p.transform_output:
        transform_output = getattr(func, 'get_batch_transform_' + p.transform_output.apply_func) \
            (*transform_parameters, **p.transform_output.params_apply)
        t_y = transform_output(y)  # transform GT labels
        y = torch.cat((y, t_y), dim=0)  # form a batch corresponding to x
    else:
        y = torch.cat((y, y), dim=0)

    # pred1 = deform.deform(pred1)
    # pred1 = transform_output(pred1)

    # save original and transformed inputs when in the debugging mode:
    if p.debug:  # and run_eagerly:
        improc.plot_batch_sample(p, x.numpy(), y.numpy(),
                                 os.path.join(p.results_path, p.exp_name, 'debug/model_input.png'))
        improc.plot_batch_sample(p, x[n:, ...].numpy(), y[n:, ...].numpy(),
                                 os.path.join(p.results_path, p.exp_name,
                                              'debug/model_transformed_input.png'))

    # -- end transformation --

    # separate differently transformed
    pred1, pred2 = pred[:n, ...], pred[n:, ...]
    if p.transform_output:
        # transform the first half of the predictions to align it with the second half:
        pred1 = transform_output(pred1)

    # separate labeled images from the rest
    yl = torch.cat((y[:n_labeled, ...], y[:n_labeled, ...]), dim=0)
    predl = torch.cat((pred[:n_labeled, ...], pred[n:(n + n_labeled), ...]), dim=0)

    # supervised loss
    loss_sup = kl_divergence(yl, predl)

    # unsupervised loss made symmetric (e.g. KL divergence is not symmetric)
    loss_usup = (kl_divergence(pred1, pred2) + kl_divergence(pred2, pred1)) / 2

    # total loss: supervised + weight * unsupervised consistency
    loss_value = loss_sup + p.alpha * loss_usup

    return loss_value, loss_sup, loss_usup, (yl, predl), (pred1, pred2)


