# import tensorflow as tf
#
# from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, \
#     GlobalAveragePooling2D, BatchNormalization, Dropout

import torch
from pyoneer_main import models
from torch import nn
from torchinfo import summary
import loss_torch


class Print(nn.Module):
    def __init__(self, label):
        super(Print, self).__init__()
        self.label = label

    def forward(self, x):
        print('label: ', self.label)
        print(x.shape)
        return x


def simple_model(input: torch.tensor) -> nn.Sequential:
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
    # NUM_CHANNELS = 3

    flat_input = torch.flatten(input)

    if activation == 'LeakyReLU':
        activation_f = nn.LeakyReLU(0.1)
    else:
        activation_f = nn.ReLU()

    # Conv layer params = (m*n*d+1)*k

    model = nn.Sequential(
                            Print(0),
                            nn.Conv2d(in_channels=3, out_channels=96,kernel_size=(3, 3)),
                            activation_f,
                            nn.BatchNorm2d(96),
                            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3)),
                            activation_f,
                            nn.BatchNorm2d(96),
                            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3)),
                            activation_f,
                            nn.BatchNorm2d(96),

                            Print(1),
                            nn.MaxPool2d(kernel_size=(2, 2)),
                            nn.Dropout2d(p=dropout),

                            Print(2),
                            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3, 3)),
                            activation_f,
                            nn.BatchNorm2d(192),
                            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3)),
                            activation_f,
                            nn.BatchNorm2d(192),
                            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3)),
                            activation_f,
                            nn.BatchNorm2d(192),

                            Print(3),
                            nn.MaxPool2d(kernel_size=(2, 2)),
                            nn.Dropout2d(p=dropout),

                            Print(4),
                            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3)),
                            activation_f,
                            nn.BatchNorm2d(192),
                            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 1)),
                            activation_f,
                            nn.BatchNorm2d(192),
                            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 1)),
                            activation_f,
                            nn.BatchNorm2d(192),

                            Print(5),
                            nn.AvgPool2d(192),

                            Print(6),
                            nn.Linear(192, 10),
                            nn.Softmax(dim=1),
    )
    # print('Instantiated a complicated model:\n' + str(model))
    return model

# summary(simple_model(torch.empty([3,32,32])))
# print(models.get_simple_model().summary())

class SemiSupervisedConsistencyModelTorch(nn.Module):

    # def __init__(self, p, optimizer, loss, metrics=[]):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, data):
        return self.model(data)




# summary(large_model(torch.empty([32,32]), '', 0.4))
# print(models.get_model_conv_small('', 0.4).summary())
