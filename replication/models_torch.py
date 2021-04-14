import torch
from pyoneer_main import models
from torch import nn
from torchinfo import summary
import loss_torch


class Print(nn.Module):
    """
    Debug layer for printing layer dimensions, useful for debugging
    """

    def __init__(self, label):
        super(Print, self).__init__()
        self.label = label

    def forward(self, x):
        # Print dimensions of the input and perform identity operation
        print('label: ', self.label, ', shape: ', x.shape)
        return x


def simple_model(input: torch.tensor) -> nn.Sequential:
    """
    :param input: a tensor which we use to define the input dimension
    :return: a simple neural network, consisting of 3 FC layers with ReLU nonlinearities and a softmax at the end.
            1,841,162 trainable parameters
    """
    flat_input = torch.flatten(input)
    model = nn.Sequential(nn.Linear(flat_input.size()[0], 512),
                          nn.ReLU(),
                          nn.Linear(512, 512, ),
                          nn.ReLU(),
                          nn.Linear(512, 10),
                          nn.Softmax(dim=1),
                          )
    print('Instantiated a simple model:\n' + str(model))
    return model


def large_model(input: torch.tensor, activation: str, dropout) -> nn.Sequential:
    """
    Instantiates a large CNN. There are 26 layers in total. The first part of the network consists of 3 stacks of
    convolutional layer with batchnorm. After this we have a max pooling layer and a dropout layer. Then there is
    another stack of 3 convolutional layers with batchnorm stack, again followed by a max pooling and dropout layer.
    Finally, there is another stack of 3 convolutional layers with batchnorm, followed by average pooling,
    a fully connected layer and a softmax nonlinearity. Note that this Pytorch model is exactly the same as the
    Tensorflow model implemented in the paper, apart from having half the trainable parameters in the batchnorm
    layers (a difference between Pytorch and Tensorflow implementations) and apart from the fact that we output
    probabilities using a softmax layer, instead of an image segmentation.
    :param input: tensor used for compatibility reasons
    :param activation: string specifying the type of activation function
    :param dropout:
    dropout probability, in [0,1] :return: a large neural network
    """

    flat_input = torch.flatten(input)

    if activation == 'LeakyReLU':
        activation_f = nn.LeakyReLU(0.1)
    else:
        activation_f = nn.ReLU()

    # Conv layer params = (m*n*d+1)*k

    model = nn.Sequential(
        # -- 3 Convolution layers --
        # input: 256 x 3 x 32 x 32
        # Print(0),
        nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(3, 3)),
        activation_f,
        nn.BatchNorm2d(96),
        nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3)),
        activation_f,
        nn.BatchNorm2d(96),
        nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3)),
        activation_f,
        nn.BatchNorm2d(96),
        # Print(1),
        # output: 256 x 96 x 26 x 26

        # -- 1 Maxpool layer --
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Dropout2d(p=dropout),
        # Print(2),
        # output: 256 x 96 x 13 x 13

        # -- 3 Convolution layers --
        nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3, 3)),
        activation_f,
        nn.BatchNorm2d(192),
        nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3)),
        activation_f,
        nn.BatchNorm2d(192),
        nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3)),
        activation_f,
        nn.BatchNorm2d(192),
        # Print(3),
        # output: 256 x 192 x 7 x 7

        # -- 1 Maxpool layer --
        nn.MaxPool2d(kernel_size=(2, 2)),
        nn.Dropout2d(p=dropout),
        # Print(4),
        # output: 256 x 192 x 3 x 3

        # -- 3 Convolution layers --
        # note: 1 time 3x3 window, 2 times 1x1 window
        nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3)),
        activation_f,
        nn.BatchNorm2d(192),
        nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 1)),
        activation_f,
        nn.BatchNorm2d(192),
        nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 1)),
        activation_f,
        nn.BatchNorm2d(192),
        # Print(5),
        # output: 256 x 192 x 1 x 1

        nn.AvgPool3d(kernel_size=(192, 1, 1)),
        # Print(6),
        # output: 256 x 1 x 1 x 1

        nn.Linear(1, 10),
        # Print(7),
        # output: 256 x 1 x 1 x 10
        nn.Softmax(dim=3),
        # Print(8),
        # output: 256 x 1 x 1 x 10
    )
    # print('Instantiated a complicated model:\n' + str(model))
    return model

class SemiSupervisedConsistencyModelTorch(nn.Module):

    # def __init__(self, p, optimizer, loss, metrics=[]):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data):
        return self.model(data)

# Statements used for debugging the architecture
# summary(simple_model(torch.empty([3,32,32])))
# print(models.get_simple_model().summary())
print('pytorch:')
summary(large_model(torch.empty([32,32]), '', 0.4))
print('tensorflow:')
print(models.get_model_conv_small('', 0.4).summary())
