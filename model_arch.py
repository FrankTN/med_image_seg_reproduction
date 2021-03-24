# import tensorflow as tf
#
# from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, \
#     GlobalAveragePooling2D, BatchNormalization, Dropout

import torch
from pyoneer_main import models
from torch import nn
from torchinfo import summary
import loss_torch


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

    # def __init__(self, p, optimizer, loss, metrics=[]):
    def __init__(self, size):
        super().__init__()
        self.model = simple_model(size)

    def forward(self, data):
        return self.model.forward(data)




summary(large_model(torch.empty([32,32]), '', 0.4))
print(models.get_model_large('', 0.4).summary())
