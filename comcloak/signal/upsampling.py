from comcloak.utils.tensors import flatten_last_dims
import torch
import torch.nn as nn

# # Flatten the last dimensions
# tensor = torch.randn(2, 3, 4, 5)
# flattened_last_dims = flatten_last_dims(tensor, 3)
# print("Flattened last dimensions:", flattened_last_dims.shape)

class Upsampling(nn.Module):
    """Upsampling(samples_per_symbol, axis=-1, **kwargs)

    Upsamples a tensor along a specified axis by inserting zeros
    between samples.

    Parameters
    ----------
    samples_per_symbol: int
        The upsampling factor. If ``samples_per_symbol`` is equal to `n`,
        then the upsampled axis will be `n`-times longer.

    axis: int
        The dimension to be up-sampled. Must not be the first dimension.

    Input
    -----
    x : [...,n,...], tf.DType
        The tensor to be upsampled. `n` is the size of the `axis` dimension.

    Output
    ------
    y : [...,n*samples_per_symbol,...], same dtype as ``x``
        The upsampled tensor.
    """
    def __init__(self, samples_per_symbol, axis=-1, **kwargs):
        
        """
        Args:
            samples_per_symbol (int): The upsampling factor.
            axis (int): The dimension to be up-sampled. Must not be the first dimension.
        """
        super().__init__(**kwargs)
        self.samples_per_symbol = samples_per_symbol
        self.axis = axis

    def forward(self, x):
        shape = list(x.shape)
        axis_length = shape[self.axis]
        new_length = axis_length * self.samples_per_symbol

        # Create a new shape with the upsampled length along the specified axis
        new_shape = shape[:self.axis] + [new_length] + shape[self.axis+1:]

        # Create an empty tensor with the new shape
        y = torch.zeros(new_shape, dtype=x.dtype, device=x.device)

        # Copy the original values into the upsampled tensor
        idx = [slice(None)] * len(shape)
        idx[self.axis] = slice(None, None, self.samples_per_symbol)
        y[idx] = x
        return y

# 示例用法
# x = torch.randn(2, 3, 4, 5)
# print("Original tensor:")
# print(x)

# # 创建上采样层
# upsample = Upsampling(samples_per_symbol=2, axis=2)
    
# # 对张量进行上采样
# y = upsample(x)
# print("\nUpsampled tensor:")
# print(y)
# print("\nUpsampled tensor shape:", y)








