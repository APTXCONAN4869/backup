from abc import ABC, abstractmethod
from comcloak.utils.tensors import expand_to_rank
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#Example usage:
# tensor = torch.tensor([1, 2, 3])
# expanded_tensor = expand_to_rank(tensor, 3, axis=1)
# print("Original tensor:", tensor)
# print("Expanded tensor:", expanded_tensor)
# print("Shape of expanded tensor:", expanded_tensor.shape)



class Window(nn.Module, ABC):
    def __init__(self, length, trainable=False, normalize=False, dtype=torch.float32,**kwargs):
        super().__init__(**kwargs)

        assert length > 0, "Length must be positive"
        self._length = length

        assert isinstance(trainable, bool), "trainable must be bool"
        self._trainable = trainable

        assert isinstance(normalize, bool), "normalize must be bool"
        self._normalize = normalize

        assert dtype in [torch.float32, torch.float64], "`dtype` must be either `torch.float32` or `torch.float64`"

        self.dtype = dtype

        self._coefficients = nn.Parameter(self._coefficients_source, requires_grad=self.trainable)

    @property
    @abstractmethod
    def _coefficients_source(self):
        pass

    @property
    def coefficients(self):
        w = self._coefficients

        if self.normalize:
            energy = torch.mean(w**2)
            w = w / torch.sqrt(energy)

        return w

    @property
    def length(self):
        return self._length

    @property
    def trainable(self):
        return self._trainable

    @property
    def normalize(self):
        return self._normalize

    def show(self, samples_per_symbol, domain="time", scale="lin"):
        assert domain in ["time", "frequency"], "Invalid domain"
        n_min = -(self.length // 2)
        n_max = n_min + self.length
        sampling_times = np.arange(n_min, n_max, dtype=np.float32)
        sampling_times /= samples_per_symbol

        if domain == "time":
            plt.figure(figsize=(12, 6))
            plt.plot(sampling_times, self.coefficients.detach().numpy())
            plt.title("Time domain")
            plt.grid()
            plt.xlabel(r"Normalized time $(t/T)$")
            plt.ylabel(r"$w(t)$")
            plt.xlim(sampling_times[0], sampling_times[-1])
        else:
            assert scale in ["lin", "db"], "Invalid scale"
            fft_size = max(1024, self.coefficients.shape[-1])
            h = np.fft.fft(self.coefficients.detach().numpy(), fft_size)
            h = np.fft.fftshift(h)
            h = np.abs(h)
            plt.figure(figsize=(12, 6))
            if scale == "db":
                h = np.maximum(h, 1e-10)
                h = 10 * np.log10(h)
                plt.ylabel(r"$|W(f)|$ (dB)")
            else:
                plt.ylabel(r"$|W(f)|$")
            f = np.linspace(-samples_per_symbol / 2,
                            samples_per_symbol / 2, fft_size)
            plt.plot(f, h)
            plt.title("Frequency domain")
            plt.grid()
            plt.xlabel(r"Normalized frequency $(f/W)$")
            plt.xlim(f[0], f[-1])

    def forward(self, x):
        x_dtype = x.dtype

        w = self.coefficients
        while len(w.shape) < len(x.shape):
            w = w.unsqueeze(0)

        if x_dtype.is_floating_point:
            y = x * w
        elif x_dtype.is_complex:
            w = w.to(torch.complex64)
            y = w * x

        return y

class CustomWindow(Window):
    def __init__(self, 
                length, 
                coefficients=None, 
                trainable=False, 
                normalize=False, 
                dtype=torch.float32,
                **kwargs):
        if coefficients is not None:
            assert len(coefficients) == length, "specified `length` does not match the one of `coefficients`"
            self._c = torch.tensor(coefficients, dtype=dtype)
        else:
            self._c = torch.randn([length], dtype=dtype,**kwargs)

        super().__init__(length, 
                         trainable, 
                         normalize, 
                         dtype,
                         **kwargs)

    @property
    def _coefficients_source(self):
        return self._c

# # Example usage
# length = 128
# coefficients = np.hamming(length)
# custom_window = CustomWindow(length=length, coefficients=coefficients, trainable=False, normalize=True, dtype=torch.float32)

# # Generate some data
# x = torch.linspace(0, 1, length, dtype=torch.float32)

# # Apply the window function
# y = custom_window(x)

# # Plot the window and the result
# custom_window.show(samples_per_symbol=length, domain="time")
# plt.show()

# print("Input:", x)
# print("Windowed Output:", y)

class HannWindow(Window):
    def __init__(self, length, trainable=False, normalize=False, dtype=torch.float32):
        super().__init__(length, trainable, normalize, dtype)

    @property
    def _coefficients_source(self):
        n = np.arange(self.length)
        coefficients = np.square(np.sin(np.pi * n / self.length))
        return torch.tensor(coefficients, dtype=self.dtype)

# # Example usage
# length = 128
# hann_window = HannWindow(length=length, trainable=False, normalize=True, dtype=torch.float32)

# # Generate some data
# x = torch.linspace(0, 1, length, dtype=torch.float32)

# # Apply the window function
# y = hann_window(x)

# # Plot the window and the result
# hann_window.show(samples_per_symbol=length, domain="time")
# plt.show()

# print("Input:", x.numpy())
# print("Windowed Output:", y.detach().numpy())

class HammingWindow(Window):
    @property
    def _coefficients_source(self):
        n = self.length
        nn = np.arange(n)
        a0 = 25. / 46.
        a1 = 1. - a0
        coefficients = a0 - a1 * np.cos(2. * np.pi * nn / n)
        return torch.tensor(coefficients, dtype=self.dtype)

# # # Example usage
# length = 128
# hamming_window = HammingWindow(length=length, trainable=False, normalize=True, dtype=torch.float32)

# # Generate some data
# x = torch.linspace(0, 1, length, dtype=torch.float32)

# # Apply the window function
# y = hamming_window(x)

# # Plot the window and the result
# hamming_window.show(samples_per_symbol=length, domain="time")
# plt.show()

# print("Input:", x.numpy())
# print("Windowed Output:", y.detach().numpy())

class BlackmanWindow(Window):

    @property
    def _coefficients_source(self):
        n = self.length
        nn = np.arange(n)
        a0 = 7938. / 18608.
        a1 = 9240. / 18608.
        a2 = 1430. / 18608.
        coefficients = a0 - a1 * np.cos(2. * np.pi * nn / n) + a2 * np.cos(4. * np.pi * nn / n)
        return torch.tensor(coefficients, dtype=self.dtype)

# # Example usage
# length = 128
# blackman_window = BlackmanWindow(length=length, trainable=False, normalize=True, dtype=torch.float32)

# # Generate some data
# x = torch.linspace(0, 1, length, dtype=torch.float32)

# # Apply the window function
# y = blackman_window(x)

# # Plot the window and the result
# blackman_window.show(samples_per_symbol=length, domain="time")
# plt.show()

# print("Input:", x.numpy())
# print("Windowed Output:", y.detach().numpy())
