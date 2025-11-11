import math
import numpy as np
import torch
import torch.nn as nn
from networks.utils.graph import Graph
from networks.utils.atgcn import unit_gcn, unit_tcn, conv_init, bn_init

class AGCN(nn.Module):
    r"""Adaptive (Spatial Temporal) Graph Convolutional Network.

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        - symmetry: Symmetry Configuration
        For more information, please refer to the thesis.
        in_channels = C (int): Number of channels in the input data

    Shape:
        - Input: :math:`(N, C, T, V)`
        - Output: :math:`(N, C, T, V)` 
        
    where
        :math:`N` is the batch size,
        :math:`C` is a length of input sequence,
        :math:`T` is the length of the input sequence,
        :math:`V` is the number of graph nodes.
    """
    def __init__(self, strategy="uniform", in_channels=3):
        super().__init__()

        # Load graph
        self.graph = Graph(strategy=strategy)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        self.data_bn = nn.BatchNorm1d(in_channels * self.A.size(1))
        self.tcn_gcn_networks = nn.ModuleList((
            TCN_GCN_unit(in_channels, 32, self.A, residual=False),
            TCN_GCN_unit(32, 64, self.A),
            TCN_GCN_unit(64, 128, self.A),
        ))

        self.fcn = nn.Conv2d(128, in_channels, kernel_size=1)
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V = x.size()

        # Data normalization
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous() # N, V, C, T
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous() # N, C, T, V

        # Forward
        for gcn in self.tcn_gcn_networks:
            x = gcn(x)

        # Features
        _, c, t, v = x.size()
        feature = x.view(N, c, t, v)

        # Prediction
        x = self.fcn(x)
        output = x.view(N, -1, t, v)

        return output, feature

class TCN_GCN_unit(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        A (torch.Tensor) : Input graph adjacency matrix in :math:`(K, V, V)` format
        stride (int, optional): Stride of the temporal convolution. Default: 1
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        
    where
        :math:`N` is the batch size,
        :math:`T_{in}/T_{out}` is the length of the input/output sequence,
        :math:`V` is the number of graph nodes.

    """
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)

