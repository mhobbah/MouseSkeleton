# The based unit of graph convolutional networks.
import torch
import torch.nn as nn
from math import sqrt

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)
    
class unit_gcn(nn.Module):
    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        A (torch.Tensor) : Input graph adjacency matrix in :math:`(K, V, V)` format
        coff_embedding (int) : Coefficient to define the dimension of the embedding space of the data. Enables the calculation of the C adjacency matrix. Default: 4
        dropout (float between 0 and 1, optional): Dropout rate of the final output. Default: 0.5
        
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        
    where
        :math:`N` is the batch size,
        :math:`T_{in}/T_{out}` is the length of the input/output sequence,
        :math:`V` is the number of graph nodes. 
    """

    def __init__(self, in_channels, out_channels, A, coff_embedding=4, dropout=0.5):
        super().__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.A = A
        self.PA = nn.Parameter(self.A.clone()) # B adjacency matrix
        nn.init.constant_(self.PA, 1e-6)
        self.num_subset = A.size(0)

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.dropout = nn.Dropout(dropout, inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        n, c, t, v = x.size()
        A = self.A.to(x.device)
        A = A + self.PA # A + B 

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(n, v, self.inter_c * t)
            A2 = self.conv_b[i](x).view(n, self.inter_c * t, v)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # (N, V, V), C adjacency matrix
            A1 = A1 + A[i] # complete adjacency matrix
            A2 = x.view(n, c * t, v)
            z = self.conv_d[i](torch.matmul(A2, A1).view(n, c, t, v))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.dropout(y)

class unit_tcn(nn.Module):
    """The basic module for applying a temporal convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the temporal convolving kernel. Default: 9
        stride (int, optional): Stride of the temporal convolution. Default: 1

    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format

    where
        :math:`N` is the batch size,
        :math:`T_{in}/T_{out}` is the length of the input/output sequence,
        :math:`V` is the number of graph nodes. 
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x