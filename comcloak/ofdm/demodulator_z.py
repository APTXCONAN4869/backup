# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for the OFDM Demodulator"""

import torch
import torch.nn as nn
import torch.fft
from numpy.fft import fftshift
import numpy as np
import torch
from comcloak.constants import PI

def insert_dims(tensor, num_dims, axis=-1):
    """Adds multiple length-one dimensions to a tensor.

    This operation is an extension to TensorFlow's `expand_dims` function.
    It inserts `num_dims` dimensions of length one starting from the
    dimension `axis` of a `tensor`. The dimension
    index follows Python indexing rules, i.e., zero-based, where a negative
    index is counted backward from the end.

    Args:
        tensor : A tensor.
        num_dims (int) : The number of dimensions to add.
        axis : The dimension index at which to expand the
               shape of `tensor`. Given a `tensor` of `D` dimensions,
               `axis` must be within the range `[-(D+1), D]` (inclusive).

    Returns:
        A tensor with the same data as `tensor`, with `num_dims` additional
        dimensions inserted at the index specified by `axis`.
    """
    assert num_dims >= 0, "`num_dims` must be nonnegative."
    rank = tensor.dim()
    assert -(rank + 1) <= axis <= rank, "`axis` is out of range `[-(D+1), D]`)"
    
    axis = axis if axis >= 0 else rank + axis + 1
    shape = list(tensor.shape)
    new_shape = shape[:axis] + [1] * num_dims + shape[axis:]
    return tensor.view(*new_shape)

def expand_to_rank(tensor, target_rank, axis=-1):
    """Inserts as many axes to a tensor as needed to achieve a desired rank.

    This operation inserts additional dimensions to a `tensor` starting at
    `axis`, so that the rank of the resulting tensor has rank
    `target_rank`. The dimension index follows Python indexing rules, i.e.,
    zero-based, where a negative index is counted backward from the end.

    Args:
        tensor : A tensor.
        target_rank (int) : The rank of the output tensor.
            If `target_rank` is smaller than the rank of `tensor`,
            the function does nothing.
        axis (int) : The dimension index at which to expand the
               shape of `tensor`. Given a `tensor` of `D` dimensions,
               `axis` must be within the range `[-(D+1), D]` (inclusive).

    Returns:
        A tensor with the same data as `tensor`, with
        `target_rank` - rank(`tensor`) additional dimensions inserted at the
        index specified by `axis`.
        If `target_rank` <= rank(`tensor`), `tensor` is returned.
    """
    num_dims = max(target_rank - tensor.dim(), 0)
    if num_dims > 0:
        tensor = insert_dims(tensor, num_dims, axis)
    return tensor


def fft(tensor, axis=-1):
    r"""Computes the normalized DFT along a specified axis.

    This operation computes the normalized one-dimensional discrete Fourier
    transform (DFT) along the ``axis`` dimension of a ``tensor``.
    For a vector :math:`\mathbf{x}\in\mathbb{C}^N`, the DFT
    :math:`\mathbf{X}\in\mathbb{C}^N` is computed as

    .. math::
        X_m = \frac{1}{\sqrt{N}}\sum_{n=0}^{N-1} x_n \exp \left\{
            -j2\pi\frac{mn}{N}\right\},\quad m=0,\dots,N-1.

    Input
    -----
    tensor : torch.complex
        Tensor of arbitrary shape.
    axis : int
        Indicates the dimension along which the DFT is taken.

    Output
    ------
    : torch.complex
        Tensor of the same dtype and shape as ``tensor``.
    """
    fft_size = tensor.shape[axis]
    scale = 1 / torch.sqrt(torch.tensor(fft_size, dtype=tensor.dtype, device=tensor.device))

    if axis not in [-1, tensor.ndim]:
        tensor = tensor.transpose(axis, -1)
        output = torch.fft.fft(tensor)
        output = output.transpose(axis, -1)
    else:
        output = torch.fft.fft(tensor)

    return scale * output



class OFDMDemodulator(nn.Module):
    """
    OFDMDemodulator(fft_size, l_min, cyclic_prefix_length, **kwargs)

    Computes the frequency-domain representation of an OFDM waveform
    with cyclic prefix removal.
    
    The demodulator assumes that the input sequence is generated by the
    :class:`~sionna.channel.TimeChannel`. For a single pair of antennas,
    the received signal sequence is given as:

    .. math::

        y_b = \sum_{\ell =L_\text{min}}^{L_\text{max}} \bar{h}_\ell x_{b-\ell} + w_b, \quad b \in[L_\text{min}, N_B+L_\text{max}-1]

    where :math:`\bar{h}_\ell` are the discrete-time channel taps,
    :math:`x_{b}` is the the transmitted signal,
    and :math:`w_\ell` Gaussian noise.

    Starting from the first symbol, the demodulator cuts the input
    sequence into pieces of size ``cyclic_prefix_length + fft_size``,
    and throws away any trailing symbols. For each piece, the cyclic
    prefix is removed and the ``fft_size``-point discrete Fourier
    transform is computed.

    Since the input sequence starts at time :math:`L_\text{min}`,
    the FFT-window has a timing offset of :math:`L_\text{min}` symbols,
    which leads to a subcarrier-dependent phase shift of
    :math:`e^{\frac{j2\pi k L_\text{min}}{N}}`, where :math:`k`
    is the subcarrier index, :math:`N` is the FFT size,
    and :math:`L_\text{min} \le 0` is the largest negative time lag of
    the discrete-time channel impulse response. This phase shift
    is removed in this layer, by explicitly multiplying
    each subcarrier by  :math:`e^{\frac{-j2\pi k L_\text{min}}{N}}`.
    This is a very important step to enable channel estimation with
    sparse pilot patterns that needs to interpolate the channel frequency
    response accross subcarriers. It also ensures that the
    channel frequency response `seen` by the time-domain channel
    is close to the :class:`~sionna.channel.OFDMChannel`.

    Parameters
    ----------
    fft_size : int
        FFT size (, i.e., the number of subcarriers).

    l_min : int
        The largest negative time lag of the discrete-time channel
        impulse response. It should be the same value as that used by the
        `cir_to_time_channel` function.

    cyclic_prefix_length : int
        Integer indicating the length of the cyclic prefix that
        is prepended to each OFDM symbol.

    Input
    -----
    :[...,num_ofdm_symbols*(fft_size+cyclic_prefix_length)+n], torch.complex64
        Tensor containing the time-domain signal along the last dimension.
        `n` is a nonnegative integer.

    Output
    ------
    :[...,num_ofdm_symbols,fft_size], torch.complex64
        Tensor containing the OFDM resource grid along the last
        two dimension.
    """

    def __init__(self, fft_size, l_min, cyclic_prefix_length=0):
        super(OFDMDemodulator, self).__init__()
        self.fft_size = fft_size
        self.l_min = l_min
        self.cyclic_prefix_length = cyclic_prefix_length

    @property
    def fft_size(self):
        return self._fft_size

    @fft_size.setter
    def fft_size(self, value):
        assert value > 0, "`fft_size` must be positive."
        self._fft_size = int(value)

    @property
    def l_min(self):
        return self._l_min

    @l_min.setter
    def l_min(self, value):
        assert value <= 0, "l_min must be nonpositive."
        self._l_min = int(value)

    @property
    def cyclic_prefix_length(self):
        return self._cyclic_prefix_length

    @cyclic_prefix_length.setter
    def cyclic_prefix_length(self, value):
        assert value >= 0, "`cyclic_prefix_length` must be nonnegative."
        self._cyclic_prefix_length = int(value)


    def forward(self, inputs):
        """Demodulate OFDM waveform onto a resource grid.

        Args:
            inputs (torch.complex64):
                `[...,num_ofdm_symbols*(fft_size+cyclic_prefix_length)]`.

        Returns:
            `torch.complex64` : The demodulated inputs of shape
            `[...,num_ofdm_symbols, fft_size]`.
        """
        tmp = -2 * PI * self.l_min \
            / self.fft_size \
            * torch.arange(self.fft_size, dtype=torch.float32)
        self._phase_compensation = torch.exp(torch.complex(torch.tensor(0.0), tmp))

        # Compute number of elements that will be truncated
        self._rest = np.mod(inputs.shape[-1],
                                self.fft_size + self.cyclic_prefix_length)

        # Compute number of full OFDM symbols to be demodulated
        self._num_ofdm_symbols = np.floor_divide(
                                    inputs.shape[-1]-self._rest,
                                    self.fft_size + self.cyclic_prefix_length)
        # Cut last samples that do not fit into an OFDM symbol
        inputs = inputs if self._rest==0 else inputs[...,:-self._rest]
        # print("demod:inputs:",inputs)
        # Reshape input to separate OFDM symbols
        new_shape = list(inputs.shape[:-1]) + [self._num_ofdm_symbols, self.fft_size + self.cyclic_prefix_length]
        
        x = inputs.view(new_shape)

        # Remove cyclic prefix
        x = x[..., self.cyclic_prefix_length:]
        # print("demod:x:",x)

        # Compute FFT
        x = fft(x)
        # print("demod:x after fft:",x)

        # Apply phase shift compensation to all subcarriers
        rot = self._phase_compensation.to(x.dtype)
        rot = expand_to_rank(rot, x.dim(), 0)
        x = x * rot

        # Shift DC subcarrier to the middle
        x = fftshift(x, axes=-1)

        return torch.tensor(x)
