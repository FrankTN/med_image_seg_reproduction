# import tensorflow as tf
#
# from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, \
#     GlobalAveragePooling2D, BatchNormalization, Dropout

import torch
from pyoneer_main import models
from torch import nn
from torchinfo import summary


def simple_model(input: torch.tensor) -> nn.Sequential:
    flat_input = torch.flatten(input)
    model = nn.Sequential(nn.Linear(flat_input.size()[0], 512),
                          nn.ReLU(),
                          nn.Linear(512, 512, ),
                          nn.ReLU(),
                          nn.Linear(512, 10),
                          nn.Softmax(),
                          )
    print('Instantiated a simple model:\n' + str(model))
    return model


def large_model(input: torch.tensor, activation_choice: str, dropout) -> nn.Sequential:
    # NUM_CHANNELS = 3

    flat_input = torch.flatten(input)

    if activation_choice == 'LeakyReLU':
        activation = nn.LeakyReLU(0.1)
    else:
        activation = nn.ReLU()

    model = nn.Sequential(
                            nn.Conv2d(in_channels=32, out_channels=96,kernel_size=(3,3)),
                            activation,
                            nn.BatchNorm2d(2*96),
                            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3)),
                            activation,
                            nn.BatchNorm2d(2*96),
                            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3)),
                            activation,
                            nn.BatchNorm2d(2*96),

                            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                            nn.Dropout2d(p=dropout),

                            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3, 3)),
                            activation,
                            nn.BatchNorm2d(2*192),
                            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3)),
                            activation,
                            nn.BatchNorm2d(2*192),
                            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3)),
                            activation,
                            nn.BatchNorm2d(2*192),

                            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                            nn.Dropout2d(p=dropout),

                            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3)),
                            activation,
                            nn.BatchNorm2d(2*192),
                            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 1)),
                            activation,
                            nn.BatchNorm2d(2*192),
                            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 1)),
                            activation,
                            nn.BatchNorm2d(2*192),

                            nn.AvgPool2d(192),

                            nn.Linear(192, 10),
                            nn.Softmax(),
    )
    print('Instantiated a complicated model:\n' + str(model))
    return model

# summary(simple_model(torch.empty([3,32,32])))
# print(models.get_simple_model().summary())

class SemiSupervisedConsistencyModelTorch(nn.Module):

    def __init__(self, p, optimizer, loss, metrics=[]):
        super().__init__()
        self.p = p
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def compute_loss(self, data):
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

    def forward(self, data):
        """
        This method is called by model.fit() for every batch.
        It should compute gradients, update model parameters and metrics.

        Parameters
        ----------
        data : tuple
            Batch received from the generator.

        Returns
        -------
        metric_values : dictionary
            Current values of all metrics (including loss terms).

        """
        # compute gradient wrt parameters
        loss_value, loss_sup, loss_usup, pair_sup, pair_usup = self.compute_loss(data)
        self.optimizer.zero_grad()
        # self.loss.backward()

        self.optimizer.step()

        # with tf.GradientTape() as tape:
        #     loss_value, loss_sup, loss_usup, pair_sup, pair_usup = self.compute_loss(data)

        grads = tape.gradient(loss_value, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        metric_values = self.update_metrics(data, [loss_value, loss_sup, loss_usup], pair_sup, pair_usup)

        return metric_values



class SemiSupervisedConsistencyModel(tf.keras.Model):

    def compile(self, p, optimizer, loss, metrics=[], run_eagerly=False):
        """
        Compile the model.

        Parameters
        ----------
        p : parameters (an OmegaConf object)
        optimizer : a keras optimizer
            A keras optimizer. See tf.keras.optimizers.
        loss : TF function
            A loss function to be used for supervised and unsupervised terms.
        metrics : a list of keras metrics, optional
            Metrics to be computed for labeled and unlabeled examples.
            See self.update_metrics to see how they are handled.
        run_eagerly : bool, optional
            If True, this Model's logic will not be wrapped in a tf.function;
            one thus can debug it more easily (e.g. print inside train_step).
            The default is False.

        Returns
        -------
        None.

        """
        super(SemiSupervisedConsistencyModel, self).compile()
        self.p = p
        self.optimizer = optimizer
        self.loss = loss
        self.loss_trackers = [tf.keras.metrics.Mean(name='loss'),
                              tf.keras.metrics.Mean(name='loss_sup'),
                              tf.keras.metrics.Mean(name='loss_usup')]
        self.extra_metrics = metrics

        self._run_eagerly = run_eagerly

    def compute_loss(self, data):
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
        n_labeled = tf.cast(tf.math.count_nonzero(labeled), tf.int32) // 2
        n = tf.shape(x)[0] // 2

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

    def update_metrics(self, data, loss_values, pair_sup, pair_usup):
        """
        Updates loss trackers and metrics so that they return the current moving averages.

        """

        # update all the loss trackers with current batch loss values
        for loss_tracker, loss_value in zip(self.loss_trackers, loss_values):
            loss_tracker.update_state(loss_value)

        # obtain prediction - target pairs
        yl, predl = pair_sup
        pred1, pred2 = pair_usup

        # for every metric type
        # sup:      metrics on the labeled subset measuring GT vs clean prediction fidelity
        # usup:     metrics on the entire batch measuring consistency
        for metric_type, y_true, y_pred in zip(['sup', 'usup'],
                                               [yl, pred1],
                                               [predl, pred2]):

            for metric in self.extra_metrics:

                # if metric name contains the type name
                if metric_type in metric.name.split('_'):
                    metric.update_state(y_true, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        """
        This method is called by model.fit() for every batch.
        It should compute gradients, update model parameters and metrics.

        Parameters
        ----------
        data : tuple
            Batch received from the generator.

        Returns
        -------
        metric_values : dictionary
            Current values of all metrics (including loss terms).

        """

        # compute gradient wrt parameters
        with tf.GradientTape() as tape:
            loss_value, loss_sup, loss_usup, pair_sup, pair_usup = self.compute_loss(data)

        grads = tape.gradient(loss_value, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        metric_values = self.update_metrics(data, [loss_value, loss_sup, loss_usup], pair_sup, pair_usup)

        return metric_values

    def test_step(self, data):
        """
        This method is called by model.fit() during the validation step
        and by model.evaluate().

        """

        loss_value, loss_sup, loss_usup, pair_sup, pair_usup = self.compute_loss(data)

        metric_values = self.update_metrics(data, [loss_value, loss_sup, loss_usup], pair_sup, pair_usup)

        return metric_values

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return self.loss_trackers + self.extra_metrics


summary(large_model(torch.empty([32,32]), '', 0.4))
print(models.get_model_large('', 0.4).summary())
