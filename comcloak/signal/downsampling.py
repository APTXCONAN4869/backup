import torch
import torch.nn as nn

class Downsampling(nn.Module):
        # pylint: disable=line-too-long
    """Downsampling(samples_per_symbol, offset=0, num_symbols=None, axis=-1, **kwargs)

    Downsamples a tensor along a specified axis by retaining one out of
    ``samples_per_symbol`` elements.

    Parameters
    ----------
    samples_per_symbol: int
        The downsampling factor. If ``samples_per_symbol`` is equal to `n`, then the
        downsampled axis will be `n`-times shorter.

    offset: int
        Defines the index of the first element to be retained.
        Defaults to zero.

    num_symbols: int
        Defines the total number of symbols to be retained after
        downsampling.
        Defaults to None (i.e., the maximum possible number).

    axis: int
        The dimension to be downsampled. Must not be the first dimension.

    Input
    -----
    x : [...,n,...], tf.DType
        The tensor to be downsampled. `n` is the size of the `axis` dimension.

    Output
    ------
    y : [...,k,...], same dtype as ``x``
        The downsampled tensor, where ``k``
        is min((``n``-``offset``)//``samples_per_symbol``, ``num_symbols``).
    """
    def __init__(self, samples_per_symbol, offset=0, num_symbols=None, axis=-1,**kwargs):

        super().__init__(**kwargs)
        self.samples_per_symbol = samples_per_symbol
        self.offset = offset
        self.num_symbols = num_symbols
        self.axis = axis

    def forward(self, x):
        # Generate index array for downsampling
        original_size = x.size(self.axis)
        indices = torch.arange(self.offset, original_size, self.samples_per_symbol)
        
        if self.num_symbols is not None:
            indices = indices[:self.num_symbols]

        # Perform downsampling along the specified axis
        slices = [slice(None)] * x.dim()
        slices[self.axis] = indices
        
        return x[slices]

# # 示例用法
#     # 生成一个随机的4维张量
# x = torch.randn(2, 3, 12, 5)
# print("Original tensor shape:", x)

#     # 创建下采样层
# downsample = Downsampling(samples_per_symbol=2, offset=1, num_symbols=1, axis=2)
    
#     # 对张量进行下采样
# y = downsample(x)
# print("Downsampled tensor shape:", y)
# print(y.shape)
