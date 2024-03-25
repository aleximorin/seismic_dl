import ConvAutoEncoder as ae

from pasedav.pasedav import loading_seismic_data as lsd
from pasedav.pasedav import utils as ut
from pasedav.pasedav import data_structures
import datetime

import obspy

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch


from time import time

import matplotlib.pyplot as plt
import matplotlib
from torch.fft import rfft

matplotlib.use('Qt5Agg')
from torch.utils.data import Dataset, DataLoader
from time import time
np.random.seed(42069)

import os


def spectral_loss(x, y, lmbda=1.0, log=False):

    mse = torch.square(x - y).mean()

    xz = torch.square(torch.abs(rfft(x))/len(x))
    yz = torch.square(torch.abs(rfft(y))/len(y))

    if log:
        xz = torch.log10(xz[:, :, 1:])
        yz = torch.log10(yz[:, :, 1:])

    spectral_mse = torch.square(xz - yz).mean()

    return mse + lmbda * spectral_mse


def transform(x):
    # scalar = torch.tensor(204194.0, dtype=torch.float)
    # x = x / scalar

    x = torch.as_tensor(x, dtype=torch.float)
    xmax = torch.max(torch.abs(x))
    x = x / xmax
    x = x - x.mean()

    return x


class SeismicDataset(Dataset):
    def __init__(self, hdf5_file, transform=None):
        """
        Args:
            hdf5_file (h5py.File): An open HDF5 file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.hdf5_file = hdf5_file
        self.transform = transform
        self.keys = list(hdf5_file.keys())

        # UGLY HACK TO REMOVE
        self.keys = self.keys[:40] + self.keys[41:]

        self.lengths = [len(hdf5_file[key]) for key in self.keys]
        self.cumulative_lengths = torch.cumsum(torch.tensor(self.lengths), 0)

    def __len__(self):
        return self.cumulative_lengths[-1].item()

    def __getitem__(self, idx):
        dataset_index = torch.searchsorted(self.cumulative_lengths, idx, right=True)
        local_idx = idx - self.cumulative_lengths[dataset_index - 1] if dataset_index > 0 else idx
        key = self.keys[dataset_index]
        data = self.hdf5_file[key][local_idx]
        if self.transform:
            data = self.transform(data)
        return data


if __name__ == '__main__':

    # autoencoder parameters
    TRACE_LENGTH = 2048
    ENCODING_DIM = 100
    KERNEL_SIZE = 3  # [11, 11, 11, 1]  # [1, 3, 3, 3]
    STRIDE = 2  # [4, 4, 4, 1]  # [1, 2, 2, 2]
    CONV_CHANNELS = [1, 8, 64, 128, 128]
    BATCH_SIZE = 64
    N_EPOCH = 50
    LR = 1e-5
    SPECTRAL_LAMBDA = 0.0  # 5e-3

    # we instantiate the different torch objects we will need
    print('creating autoencoder')
    model = ae.ConvAutoEncoder(trace_length=TRACE_LENGTH,
                               encoding_dim=ENCODING_DIM,
                               kernel_size=KERNEL_SIZE,
                               conv_channels=CONV_CHANNELS,
                               stride=STRIDE)
    print(model.encoder.out_shapes)

    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    model.train()

    # if we mean to load
    model_path = 'conv_fft.p'
    state = torch.load(model_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

    # the different time indices are akin to the number of epochs
    try:
        epoch_losses = state['loss']
    except KeyError:
        epoch_losses = []

    import h5py
    from torch.utils.data import DataLoader

    path = r'F:\ae_data\training_data.h5py'

    with h5py.File(path, 'r') as file:
        sd = SeismicDataset(file, transform=transform)
        loader = DataLoader(sd, batch_size=BATCH_SIZE, shuffle=True)

        # Example: Run training loop
        for epoch in range(N_EPOCH):
            print(f'epoch number {epoch + 1} out of {N_EPOCH}')
            loader_loss = []

            encoded = []
            for i, x in enumerate(loader):

                y = model(x)

                loss = spectral_loss(x, y, lmbda=SPECTRAL_LAMBDA)

                if torch.isnan(loss):
                    print()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loader_loss.append(loss.detach().numpy())

                print(f'\rbatch {i + 1} out of {len(sd) // BATCH_SIZE + 1}, '
                      f'current loss: {loader_loss[-1]:.6f}, '
                      f'median loss: {np.median(loader_loss):.6f}, '
                      f'mean loss: {np.mean(loader_loss):.6f}, '
                      f'loss std: {np.std(loader_loss):.6f}', end='')

                if i % 500 == 0:
                    state = dict(state_dict=model.state_dict(),
                                 optimizer=optimizer.state_dict(),
                                 loss=[*epoch_losses, loader_loss])
                    torch.save(state, model_path)

            epoch_losses.append(loader_loss)
            state = dict(state_dict=model.state_dict(),
                         optimizer=optimizer.state_dict(),
                         loss=[*epoch_losses, loader_loss])
            try:
                torch.save(state, model_path)
            except AttributeError:
                pass
            print()



"""
from scipy.signal import periodogram

fig, ax = plt.subplots()

offset = 2

for i in range(len(x)):
    ax.plot(x[i].T + i * offset, c='k', lw=0.5)
    ax.plot(y[i].T.detach() + i * offset, c='tab:red', lw=0.5)
    ax.plot([0, x.size(-1)], [i * offset, i * offset], c='k', ls='dashed', lw=0.25)

fig, ax = plt.subplots()
for i, trace in enumerate([x, y.detach()]):
    f, z = periodogram(trace, axis=-1)
    ax.plot(1/f, z[:, 0].T, c=['k', 'r'][i], lw=0.5)  # , alpha=0.5)
    
ax.set_xlim(0, 100)
    
plt.figure()
for i, l in enumerate(epoch_losses):
    plt.semilogy(np.arange(len(l)) + i * len(l), l)

plt.plot(np.arange(len(loader_loss)) + (i + 1) * len(loader), loader_loss)
"""