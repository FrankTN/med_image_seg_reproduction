# TF functions (losses, metrics etc.)

from collections.abc import Iterable

import elasticdeform.torch as etorch

#%% Transform ops
import torch


def get_batch_transform_identity(displacement):
    """
    Returns an identity mapping. Merely for testing purposes.

    """
    return lambda x: x

# add a given noise tensor and clip so that image values are not outside the normal range:
def get_batch_transform_add_noise(noise):
    """
    Add a given noise tensor and clip the result so that image value sare not outside the normal range.

    """
    return lambda x: torch.clip(x + noise, torch.min(x), torch.max(x))

# apply any displacement map using elasticdeform library:
def get_batch_transform_displacement_map(displacement, interpolation_order = 3):
    """
    Apply a displacement map using elasticdeform library.

    """
    
    fn = lambda x: (etorch.deform_grid(x[0], x[1], order = interpolation_order, axis = (1, 2)), x[1])
    
    return lambda x: tf.map_fn(fn, elems = (x, displacement))[0]

