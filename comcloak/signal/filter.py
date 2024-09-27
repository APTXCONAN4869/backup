from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from . import window as win
import torch.nn as nn
from . import empirical_aclr , convolve
import torch

class Filter(nn.Module,ABC):
    # pylint: disable=line-too-long
    r"""Filter(span_in_symbols, samples_per_symbol, window=None, normalize=True, trainable=False, dtype=tf.float32, **kwargs)

    This is an abtract class for defining a filter of ``length`` K which can be
    applied to an input ``x`` of length N.

    The filter length K is equal to the filter span in symbols (``span_in_symbols``)
    multiplied by the oversampling factor (``samples_per_symbol``).
    If this product is even, a value of one will be added.

    The filter is applied through discrete convolution.

    An optional windowing function ``window`` can be applied to the filter.

    The `dtype` of the output is `tf.float` if both ``x`` and the filter coefficients have dtype `tf.float`.
    Otherwise, the dtype of the output is `tf.complex`.

    Three padding modes are available for applying the filter:

    *   "full" (default): Returns the convolution at each point of overlap between ``x`` and the filter.
        The length of the output is N + K - 1. Zero-padding of the input ``x`` is performed to
        compute the convolution at the borders.
    *   "same": Returns an output of the same length as the input ``x``. The convolution is computed such
        that the coefficients of the input ``x`` are centered on the coefficient of the filter with index
        (K-1)/2. Zero-padding of the input signal is performed to compute the convolution at the borders.
    *   "valid": Returns the convolution only at points where ``x`` and the filter completely overlap.
        The length of the output is N - K + 1.

    Parameters
    ----------
    span_in_symbols: int
        Filter span as measured by the number of symbols.

    samples_per_symbol: int
        Number of samples per symbol, i.e., the oversampling factor.

    window: Window or string (["hann", "hamming", "blackman"])
        Instance of :class:`~sionna.signal.Window` that is applied to the filter coefficients.
        Alternatively, a string indicating the window name can be provided. In this case,
        the chosen window will be instantiated with the default parameters. Custom windows
        must be provided as instance.

    normalize: bool
        If `True`, the filter is normalized to have unit power.
        Defaults to `True`.

    trainable: bool
        If `True`, the filter coefficients are trainable.
        Defaults to `False`.

    dtype: tf.DType
        The `dtype` of the filter coefficients.
        Defaults to `tf.float32`.

    Input
    -----
    x : [..., N], tf.complex or tf.float
        The input to which the filter is applied.
        The filter is applied along the last dimension.

    padding : string (["full", "valid", "same"])
        Padding mode for convolving ``x`` and the filter.
        Must be one of "full", "valid", or "same". Case insensitive.
        Defaults to "full".

    conjugate : bool
        If `True`, the complex conjugate of the filter is applied.
        Defaults to `False`.

    Output
    ------
    y : [...,M], tf.complex or tf.float
        Filtered input.
        It is `tf.float` only if both ``x`` and the filter are `tf.float`.
        It is `tf.complex` otherwise.
        The length M depends on the ``padding``.
    """
    def __init__(self,
                 span_in_symbols,
                 samples_per_symbol,
                 window=None,
                 normalize=True,
                 trainable=False,
                 dtype= torch.float32,
                 **kwargs):
        super().__init__(**kwargs)

        assert span_in_symbols>0, "span_in_symbols must be positive"
        self._span_in_symbols = span_in_symbols

        assert samples_per_symbol>0, "samples_per_symbol must be positive"
        self._samples_per_symbol = samples_per_symbol

        self.window = window

        assert isinstance(normalize, bool), "normalize must be bool"
        self._normalize = normalize

        assert isinstance(trainable, bool), "trainable must be bool"
        self._trainable = trainable

        # assert self.length==(self._coefficients_source.shape[-1]), \
        # "The number of coefficients must match the filter length."
        self.dtype = dtype
        if dtype in (torch.float32,torch.float64):
            self._coefficients = nn.Parameter(self._coefficients_source)
        else: 
        #dtype in (torch.complex64,torch.complex128):
            c = self._coefficients_source
            self._coefficients = [nn.Parameter(c.real),nn.Parameter(c.imag)]

    @property
    def length(self):
        """The filter length in samples"""
        l = self._span_in_symbols*self._samples_per_symbol
        l = 2*(l//2)+1 # Force length to be the next odd number
        return l

    @property
    def window(self):
        """The window function that is applied to the filter coefficients. `None` if no window is applied."""
        return self._window

    @window.setter
    def window(self, value):
        if isinstance(value, str):
            if value=="hann":
                self._window = win.HannWindow(self.length)
            elif value=="hamming":
                self._window = win.HammingWindow(self.length)
            elif value=="blackman":
                self._window = win.BlackmanWindow(self.length)
            else:
                raise AssertionError("Invalid window type")
        elif isinstance(value, win.Window) or value is None:
            self._window = value
        else:
            raise AssertionError("Invalid window type")

    @property
    def normalize(self):
        """`True` if the filter is normalized to have unit power. `False` otherwise."""
        return self._normalize

    @property
    def trainable(self):
        """`True` if the filter coefficients are trainable. `False` otherwise."""
        return self._trainable

    @property
    @abstractmethod
    def _coefficients_source(self):
        """Internal property that returns the (unormalized) filter coefficients.
        Concrete classes that inherits from this one must implement this
        property."""
        pass

    @property
    def coefficients(self):
        """The filter coefficients (after normalization)"""
        h = self._coefficients
        dtype = self.dtype

        # Combine both real dimensions to complex if necessary
        if dtype in (torch.complex64,torch.complex128):
            h = torch.complex(h[0], h[1])

        # Apply window
        if self.window is not None:
            h = self._window(h)

        # Ensure unit L2-norm of the coefficients
        if self.normalize:
            energy = torch.sum(torch.square(torch.abs(h)))
            energy = energy.to(h.dtype)
            h = h / torch.sqrt(energy)
        return h

    @property
    def sampling_times(self):
        """Sampling times in multiples of the symbol duration"""
        n_min = -(self.length//2)
        n_max = n_min + self.length
        t = np.arange(n_min, n_max, dtype=np.float32)
        t /= self._samples_per_symbol
        return t

    def show(self, response="impulse", scale="lin"):
        r"""Plot the impulse or magnitude response

        Plots the impulse response (time domain) or magnitude response
        (frequency domain) of the filter.

        For the computation of the magnitude response, a minimum DFT size
        of 1024 is assumed which is obtained through zero padding of
        the filter coefficients in the time domain.

        Input
        -----
        response: str, one of ["impulse", "magnitude"]
            The desired response type.
            Defaults to "impulse"

        scale: str, one of ["lin", "db"]
            The y-scale of the magnitude response.
            Can be "lin" (i.e., linear) or "db" (, i.e., Decibel).
            Defaults to "lin".
        """
        assert response in ["impulse", "magnitude"], "Invalid response"
        if response=="impulse":
            dtype = self.dtype
            plt.figure(figsize=(12,6))
            plt.plot(self.sampling_times, np.real(self.coefficients.detach().numpy()))
            if dtype in (torch.complex64,torch.complex128):
                plt.plot(self.sampling_times, np.imag(self.coefficients.detach().numpy()))
                plt.legend(["Real part", "Imaginary part"])
            plt.title("Impulse response")
            plt.grid()
            plt.xlabel(r"Normalized time $(t/T)$")
            plt.ylabel(r"$h(t)$")
            plt.xlim(self.sampling_times[0], self.sampling_times[-1])

        else:
            assert scale in ["lin", "db"], "Invalid scale"
            fft_size = max(1024, self.coefficients.shape[-1])
            h = np.fft.fft(self.coefficients.detach().numpy(), fft_size)
            h = np.fft.fftshift(h)
            h = np.abs(h)
            plt.figure(figsize=(12,6))
            if scale=="db":
                h = np.maximum(h, 1e-10)
                h = 10*np.log10(h)
                plt.ylabel(r"$|H(f)|$ (dB)")
            else:
                plt.ylabel(r"$|H(f)|$")
            f = np.linspace(-self._samples_per_symbol/2,
                            self._samples_per_symbol/2, fft_size)
            plt.plot(f, h)
            plt.title("Magnitude response")
            plt.grid()
            plt.xlabel(r"Normalized frequency $(f/W)$")
            plt.xlim(f[0], f[-1])

    @property
    def aclr(self):
        """ACLR of the filter

        This ACLR corresponds to what one would obtain from using
        this filter as pulse shaping filter on an i.i.d. sequence of symbols.
        The in-band is assumed to range from [-0.5, 0.5] in normalized
        frequency.
        """
        fft_size = 1024
        n = fft_size - self.coefficients.shape[-1]
        z = torch.zeros([n], self.coefficients.dtype)
        c = torch.cat([self.coefficients, z], dim=-1)
        c = c.to(torch.complex64)
        return empirical_aclr(c, self._samples_per_symbol)

    def forward(self, x, padding='full', conjugate=False):
        h = self.coefficients
        dtype = self.dtype
        if conjugate and dtype in(torch.complex64,torch.complex128):
            h = torch.conj(h)
        y = convolve(x,h,padding)
        return y
# _coefficients_source = torch.tensor([1+1j,2+2j,3+3j,4+4j])
# c = _coefficients_source
# _coefficients = [  nn.Parameter(c.real),
#                     nn.Parameter(c.imag)]      
# print(_coefficients) 

class RaisedCosineFilter(Filter):
    # pylint: disable=line-too-long
    def __init__(self,
                 span_in_symbols,
                 samples_per_symbol,
                 beta,
                 window=None,
                 normalize=True,
                 trainable=False,
                 dtype=torch.float32,
                 **kwargs):

        assert 0 <= beta <= 1, "beta must be from the intervall [0,1]"
        self._beta = beta

        super().__init__(span_in_symbols,
                         samples_per_symbol,
                         window=window,
                         normalize=normalize,
                         trainable=trainable,
                         dtype=dtype,
                         **kwargs)

    @property
    def beta(self):
        """Roll-off factor"""
        return self._beta

    @property
    def _coefficients_source(self):
        h = self._raised_cosine(self.sampling_times,
                                1.0,
                                self.beta)
        h = torch.tensor(h, dtype=self.dtype)
        return h

    def _raised_cosine(self, t, symbol_duration, beta):
        """Raised-cosine filter from Wikipedia
        https://en.wikipedia.org/wiki/Raised-cosine_filter"""
        h = np.zeros([len(t)], np.float32)
        for i, tt in enumerate(t):
            tt = np.abs(tt)
            if beta>0 and (tt-np.abs(symbol_duration/2/beta)==0):
                h[i] = np.pi/4/symbol_duration*np.sinc(1/2/beta)
            else:
                h[i] = 1./symbol_duration*np.sinc(tt/symbol_duration)\
                    * np.cos(np.pi*beta*tt/symbol_duration)\
                    /(1-(2*beta*tt/symbol_duration)**2)
        return h
#示例代码
example_filter = RaisedCosineFilter(span_in_symbols=4, samples_per_symbol=2, beta=0.25, window=None, normalize=True, trainable=False, dtype=torch.float32)
x = torch.randn([10,20], dtype=torch.float32)
y = example_filter(x, padding='same', conjugate=False)

print("Input signal: ", x)
print("Filtered signal: ", y)

example_filter.show(response="impulse")
example_filter.show(response="magnitude", scale="db")
#示例代码
samples_per_symbol = 4
span_in_symbols = 2
w = win.HannWindow(length=samples_per_symbol*span_in_symbols+1, dtype=torch.float32)
example_filter = RaisedCosineFilter(span_in_symbols=4, samples_per_symbol=2, beta=0.25, window=w, normalize=True, trainable=False, dtype=torch.float32)
x = torch.randn([10,20], dtype=torch.float32)
y = example_filter(x, padding='same')

print("Input signal: ", x)
print("Filtered signal: ", y)

example_filter.show(response="impulse")
example_filter.show(response="magnitude", scale="db")