import os
import math
import numpy
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F


class LinearReLuDropout(nn.Module):
    """A simple feedforward neural network with one hidden layer.

    One hidden layer with ReLU activation function and one output layer.

    Attributes:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of hidden units.
        output_dim (int): The number of output units.
        fc1 (torch.nn.Linear): The first fully connected layer.
        relu (torch.nn.ReLU): The ReLU activation function.
        dropout (torch.nn.Dropout): The dropout layer.
    """

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(LinearReLuDropout, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x