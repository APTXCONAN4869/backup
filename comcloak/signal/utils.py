import numpy as np
import matplotlib.pyplot as plt
from comcloak.utils.tensors import expand_to_rank
import torch
import torch.nn.functional as F


def convolve(inp, ker, padding='full', axis=-1):
    """
    Filters an input ``inp`` of length `N` by convolving it with a kernel ``ker`` of length `K`.

    The length of the kernel ``ker`` must not be greater than the one of the input sequence ``inp``.

    The `dtype` of the output is `torch.float` only if both ``inp`` and ``ker`` are `torch.float`. It is `torch.complex` otherwise.
    ``inp`` and ``ker`` must have the same precision.

    Three padding modes are available:

    *   "full" (default): Returns the convolution at each point of overlap between ``ker`` and ``inp``.
        The length of the output is `N + K - 1`. Zero-padding of the input ``inp`` is performed to
        compute the convolution at the border points.
    *   "same": Returns an output of the same length as the input ``inp``. The convolution is computed such
        that the coefficients of the input ``inp`` are centered on the coefficient of the kernel ``ker`` with index
        ``(K-1)/2`` for kernels of odd length, and ``K/2 - 1`` for kernels of even length.
        Zero-padding of the input signal is performed to compute the convolution at the border points.
    *   "valid": Returns the convolution only at points where ``inp`` and ``ker`` completely overlap.
        The length of the output is `N - K + 1`.

    Input
    ------
    inp : [...,N], torch.complex or torch.float
        Input to filter.

    ker : [K], torch.complex or torch.float
        Kernel of the convolution.

    padding : string
        Padding mode. Must be one of "full", "valid", or "same". Case insensitive.
        Defaults to "full".

    axis : int
        Axis along which to perform the convolution.
        Defaults to `-1`.

    Output
    -------
    out : [...,M], torch.complex or torch.float
        Convolution output.
        It is `torch.float` only if both ``inp`` and ``ker`` are `torch.float`. It is `torch.complex` otherwise.
        The length `M` of the output depends on the ``padding``.
    """

    # We don't want to be sensitive to case
    padding = padding.lower()
    assert padding in ('valid', 'same', 'full'), "Invalid padding method"

    # Ensure we process along the axis requested by the user
    inp = inp.transpose(axis, -1)

    # Reshape the input to a 2D tensor
    batch_shape = inp.shape[:-1]
    inp_len = inp.shape[-1]
    inp_dtype = inp.dtype
    ker_dtype = ker.dtype
    inp = inp.reshape(-1, 1, inp_len)

    # Flip the kernel
    ker = torch.flip(ker, dims=[0])
    # Reshape the kernel to 3D tensor
    ker = expand_to_rank(ker, 3, 0)

    # Pad the kernel or input if required depending on the convolution type
    if padding == 'valid':
        # No padding required in this case
        pad = 0
    elif padding == 'same':
        ker = F.pad(ker, (0, 1))
        pad = 'same'
    elif padding == 'full':
        ker_len = ker.shape[2]
        if (ker_len % 2) == 0:
            extra_padding_left = ker_len // 2
            extra_padding_right = extra_padding_left - 1
        else:
            extra_padding_left = (ker_len - 1) // 2
            extra_padding_right = extra_padding_left
        inp = F.pad(inp, (extra_padding_left, extra_padding_right))
        pad = 'same'

  # Initialize real and imaginary parts
    inp_real = inp
    ker_real = ker
    inp_imag = None
    ker_imag = None

    # Extract real and imaginary components if necessary
    if torch.is_complex(inp):
        inp_real = inp.real
        inp_imag = inp.imag
    if torch.is_complex(ker):
        ker_real = ker.real
        ker_imag = ker.imag
    # Compute convolution
    complex_output = False
    out_1 = F.conv1d(inp_real, ker_real, padding=pad)
    if torch.is_complex(inp):
        out_4 = F.conv1d(inp_imag, ker_real, padding=pad)
        complex_output = True
    else:
        out_4 = torch.zeros_like(out_1)
    if torch.is_complex(ker):
        out_3 = F.conv1d(inp_real, ker_imag, padding=pad)
        complex_output = True
    else:
        out_3 = torch.zeros_like(out_1)
    if torch.is_complex(inp) and torch.is_complex(ker):
        out_2 = F.conv1d(inp_imag, ker_imag, padding=pad)
    else:
        out_2 = torch.zeros_like(out_1)
    if complex_output:
        out = torch.complex(out_1 - out_2, out_3 + out_4)
    else:
        out = out_1

    # Reshape the output to the expected shape
    out_len = out.shape[-1]
    out = out.view(*batch_shape, out_len)
    out = out.transpose(axis, -1)

    return out

# 示例测试用例
inp = torch.rand(64, 100, dtype=torch.float32)
ker = torch.rand(10, dtype=torch.float32)
out = convolve(inp, ker, padding='full')
print(out.shape)


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
    tensor : torch.Tensor
        Tensor of arbitrary shape.
    axis : int
        Indicates the dimension along which the DFT is taken.

    Output
    ------
    : torch.Tensor
        Tensor of the same dtype and shape as ``tensor``.
    """
    N = tensor.size(axis)
    norm_factor = torch.sqrt(torch.tensor(N, dtype=tensor.dtype, device=tensor.device))
    result = torch.fft.fft(tensor, dim=axis) / norm_factor
    return result

# # 示例
# tensor = torch.randn([10,20,3],dtype = torch.complex64)
# result = fft(tensor)
# print(result)
def ifft(tensor, axis=-1):
    r"""Computes the normalized IDFT along a specified axis.

    This operation computes the normalized one-dimensional discrete inverse
    Fourier transform (IDFT) along the ``axis`` dimension of a ``tensor``.
    For a vector :math:`\mathbf{X}\in\mathbb{C}^N`, the IDFT
    :math:`\mathbf{x}\in\mathbb{C}^N` is computed as

    .. math::
        x_n = \frac{1}{\sqrt{N}}\sum_{m=0}^{N-1} X_m \exp \left\{
            j2\pi\frac{mn}{N}\right\},\quad n=0,\dots,N-1.

    Input
    -----
    tensor : torch.Tensor
        Tensor of arbitrary shape.
    axis : int
        Indicates the dimension along which the IDFT is taken.

    Output
    ------
    : torch.Tensor
        Tensor of the same dtype and shape as ``tensor``.
    """
    N = tensor.size(axis)
    norm_factor = torch.sqrt(torch.tensor(N, dtype=tensor.dtype, device=tensor.device))
    result = torch.fft.ifft(tensor, dim=axis) * norm_factor
    return result

# 示例
# tensor = torch.tensor([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j, 4.0 + 4.0j], dtype=torch.complex64)
# result = ifft(tensor)
# print(result)

def empirical_psd(x, show=True, oversampling=1.0, ylim=(-30, 3)):
    r"""Computes the empirical power spectral density.

    Computes the empirical power spectral density (PSD) of tensor ``x``
    along the last dimension by averaging over all other dimensions.
    Note that this function
    simply returns the averaged absolute squared discrete Fourier
    spectrum of ``x``.

    Input
    -----
    x : [...,N], torch.Tensor (complex)
        The signal of which to compute the PSD.

    show : bool
        Indicates if a plot of the PSD should be generated.
        Defaults to True.

    oversampling : float
        The oversampling factor. Defaults to 1.

    ylim : tuple of floats
        The limits of the y axis. Defaults to [-30, 3].
        Only relevant if ``show`` is True.

    Output
    ------
    freqs : [N], float
        The normalized frequencies at which the PSD was evaluated.

    psd : [N], float
        The PSD.
    """
    freqs = 1
    # Compute PSD
    
    x = torch.cat((x.unsqueeze(0), x.unsqueeze(0)),dim = 0)
    psd = torch.abs(fft(x))**2
    psd= psd.mean(tuple(range(x.ndim - 1)))

    psd = torch.fft.fftshift(psd)

    f_min = -0.5*oversampling
    f_max = -f_min
    freqs = torch.linspace(f_min, f_max, psd.shape[0])
  
    return (freqs , psd)
    
# # 示例
# x = torch.tensor([[1+1j,3+1j,5+1j,7+2j]] ,dtype=torch.complex64)
# freqs,psd = empirical_psd(x, show=True)
# print(freqs)
# print(psd)

def empirical_aclr(x, oversampling=1.0, f_min=-0.5, f_max=0.5):
    r"""Computes the empirical ACLR.

    Computes the empirical adjacent channel leakgae ration (ACLR)
    of tensor ``x`` based on its empirical power spectral density (PSD)
    which is computed along the last dimension by averaging over
    all other dimensions.

    It is assumed that the in-band ranges from [``f_min``, ``f_max``] in
    normalized frequency. The ACLR is then defined as

    .. math::

        \text{ACLR} = \frac{P_\text{out}}{P_\text{in}}

    where :math:`P_\text{in}` and :math:`P_\text{out}` are the in-band
    and out-of-band power, respectively.

    Input
    -----
    x : [...,N],  complex
        The signal for which to compute the ACLR.

    oversampling : float
        The oversampling factor. Defaults to 1.

    f_min : float
        The lower border of the in-band in normalized frequency.
        Defaults to -0.5.

    f_max : float
        The upper border of the in-band in normalized frequency.
        Defaults to 0.5.

    Output
    ------
    aclr : float
        The ACLR in linear scale.
    """
    freqs, psd = empirical_psd(x, oversampling=oversampling, show=False)
    ind_out = torch.where((freqs < f_min) | (freqs > f_max))[0]
    ind_in = torch.where((freqs > f_min) & (freqs < f_max))[0]


    # Compute in-band and out-of-band power
    p_out = torch.sum(psd[ind_out])
    p_in = torch.sum(psd[ind_in])


    # Compute ACLR
    aclr = p_out / p_in

    return aclr.item()

# x = torch.tensor([[1,2,5,6,4]],dtype= torch.complex64)

# # Convert to complex tensor for PyTorch
# x = torch.tensor(x, dtype=torch.cfloat)

# aclr_value = empirical_aclr(x)
# print(f"ACLR: {aclr_value}")
