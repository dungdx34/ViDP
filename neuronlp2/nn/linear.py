__author__ = 'Dung Doan'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class BiLLinear(nn.Module):
    def __init__(self, left_features, right_features, out_features, bias=True):
        super(BiLLinear, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features

        self.U = Parameter(torch.Tensor(self.out_features, self.left_features, self.right_features))
        self.W_l = Parameter(torch.Tensor(self.out_features, self.left_features))
        self.W_r = Parameter(torch.Tensor(self.out_features, self.left_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.constant_(self.bias, 0.)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left, input_right):
        left_size = input_left.size()
        right_size = input_right.size()

        batch = int(np.prod(left_size[:-1]))

        input_left = input_left.view(batch, self.left_features)
        input_right = input_right.view(batch, self.right_features)

        output = F.bilinear(input_left, input_right, self.U, self.bias)
        output = output + F.linear(input_left, self.W_l, None) + F.linear(input_right, self.W_r, None)

        return output.view(left_size[:-1] + (self.out_features, ))

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in1_features=' + str(self.left_features) \
            + ', in2_features=' + str(self.right_features) \
            + ', out_features=' + str(self.out_features) + ')'

