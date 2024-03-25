import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from torch.nn import MSELoss

import matplotlib

matplotlib.use('Qt5Agg')

device = torch.device('cpu')


def conv1d_outshape(in_length, out_channels, kernel_size, padding=0, dilation=1, stride=1):
    n_batch, in_channels, n_in = in_length
    n_out = np.floor((n_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    return n_batch, out_channels, n_out.astype(int)


def conv_transpose1d_outpadding(in_length, out_length, kernel_size, stride=1, padding=0, dilation=1):
    output_padding = out_length - (in_length - 1) * stride + 2 * padding - dilation * (kernel_size - 1) - 1
    return output_padding


def conv_transpose_outshape(in_length, out_padding, kernel_size, stride=1, padding=0, dilation=1):
    out_length = (in_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + out_padding
    return out_length


class ConvEncoder(torch.nn.Module):
    def __init__(self,
                 trace_length,
                 encoding_dim,
                 kernel_size,
                 stride,
                 conv_channels,
                 act_func,
                 padding=0,
                 dropout=0.0):

        super().__init__()
        self.trace_length = trace_length
        self.encoding_dim = encoding_dim
        self.conv_channels = conv_channels

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.act_func = act_func
        self.dropout = dropout

        self.out_shapes = []
        self.net = self._build_network()

    def _build_network(self):
        layers = []
        out_shape = (1, self.conv_channels[0], self.trace_length)
        self.out_shapes.append(out_shape)

        for i in range(len(self.conv_channels) - 1):
            in_channels = self.conv_channels[i]
            out_channels = self.conv_channels[i + 1]

            kernel_size = self.kernel_size[i]
            padding = kernel_size // 2
            stride = self.stride[i]

            conv = nn.Conv1d(in_channels=in_channels,
                             out_channels=out_channels,
                             stride=1,
                             kernel_size=kernel_size,
                             padding=padding)

            layers.append(conv)
            # layers.append(nn.BatchNorm1d(out_channels))
            layers.append(self.act_func)
            layers.append(nn.MaxPool1d(stride, stride=stride))

            # layers.append(nn.Dropout1d(p=self.dropout))

            out_shape = conv1d_outshape(out_shape, out_channels,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        stride=stride)

            self.out_shapes.append(out_shape)

        linear_dim = int(np.prod(out_shape))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(linear_dim, self.encoding_dim))

        net = nn.Sequential(*layers)
        return net

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
            # print(layer.__class__.__name__, x.shape)
        return x


class ConvDecoder(torch.nn.Module):
    def __init__(self,
                 trace_length,
                 encoding_dim,
                 kernel_size,
                 stride,
                 conv_channels,
                 encoder_shapes,
                 act_func,
                 dropout=0.0):

        super().__init__()
        self.trace_length = trace_length
        self.encoding_dim = encoding_dim
        self.conv_channels = conv_channels
        self.encoder_shapes = encoder_shapes
        self.linear_dim = int(np.prod(self.encoder_shapes[0]))

        self.kernel_size = kernel_size
        self.stride = stride
        self.act_func = act_func
        self.dropout = dropout

        self.linear = nn.Sequential(
            nn.Linear(self.encoding_dim, self.linear_dim),
            self.act_func)

        self.tanh = nn.Tanh()

        self.net = self._build_network()

    def _build_network(self):
        layers = []

        for i in range(len(self.conv_channels) - 1):
            in_channels = self.conv_channels[i]
            out_channels = self.conv_channels[i + 1]

            kernel_size = self.kernel_size[i]
            padding = kernel_size // 2
            stride = self.stride[i]

            out_padding = conv_transpose1d_outpadding(self.encoder_shapes[i][-1],
                                                      self.encoder_shapes[i + 1][-1],
                                                      kernel_size, stride, padding)

            conv = nn.ConvTranspose1d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      stride=stride,
                                      kernel_size=kernel_size,
                                      padding=padding,
                                      output_padding=out_padding)

            if i != 0:
                layers.append(self.act_func)

            layers.append(conv)
            # layers.append(nn.Dropout1d(p=self.dropout))
            # layers.append(nn.BatchNorm1d(out_channels))

        net = nn.Sequential(*layers)
        return net

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, *self.encoder_shapes[0][1:])
        for layer in self.net:
            x = layer(x)
            # print(layer.__class__.__name__, x.shape)
        #x = self.tanh(x)

        return x


class ConvAutoEncoder(torch.nn.Module):
    def __init__(self,
                 trace_length,
                 encoding_dim,
                 kernel_size,
                 conv_channels,
                 stride,
                 act_func=nn.ReLU(),
                 dropout=0.0):

        super().__init__()
        self.trace_length = trace_length
        self.encoding_dim = encoding_dim
        self.conv_channels = conv_channels

        self.kernel_size = self._assert_length(kernel_size)
        self.stride = self._assert_length(stride)
        self.act_func = act_func
        self.dropout = dropout

        self.encoder = ConvEncoder(trace_length=self.trace_length,
                                   encoding_dim=self.encoding_dim,
                                   kernel_size=self.kernel_size,
                                   conv_channels=self.conv_channels,
                                   act_func=self.act_func,
                                   stride=self.stride,
                                   dropout=self.dropout)

        self.decoder = ConvDecoder(trace_length=self.trace_length,
                                   encoding_dim=self.encoding_dim,
                                   kernel_size=self.kernel_size[::-1],
                                   conv_channels=self.conv_channels[::-1],
                                   encoder_shapes=self.encoder.out_shapes[::-1],
                                   act_func=self.act_func,
                                   stride=self.stride[::-1],
                                   dropout=self.dropout)

    def _assert_length(self, x):

        try:
            length = len(x)
            assert length == len(self.conv_channels) - 1, \
                ValueError('The provided length is not the same as the number of convolutional layers')

        except TypeError:
            x = [x for i in range(len(self.conv_channels) - 1)]

        return x

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == '__main__':
    x = torch.tensor(np.random.rand(32, 1, 2048), dtype=torch.float)

    # autoencoder parameters
    TRACE_LENGTH = 2048
    ENCODING_DIM = 10
    KERNEL_SIZE = 3
    STRIDE = 4
    CONV_CHANNELS = [1, 8, 64, 128]
    BATCH_SIZE = 32
    OVERLAP = 4

    # we instantiate the different torch objects we will need
    print('creating autoencoder')
    model = ConvAutoEncoder(trace_length=TRACE_LENGTH,
                            encoding_dim=ENCODING_DIM,
                            kernel_size=KERNEL_SIZE,
                            conv_channels=CONV_CHANNELS,
                            stride=STRIDE)

    print(model.encoder.out_shapes)


    out = model(x)

    print(out.shape)
