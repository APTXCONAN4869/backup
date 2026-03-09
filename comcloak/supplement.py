import torch
import numpy as np
from typing import Callable, List
# def gather_pytorch(input_data, indices=None, batch_dims=0, axis=0):
#     input_data = torch.tensor(input_data)
#     indices = torch.tensor(indices)
#     if axis == None:
#         axis =0 
#     if batch_dims == 0:
#         if axis < 0:
#             axis = len(input_data.shape) + axis
#         data = torch.index_select(input_data, axis, indices.flatten())
#         shape_input = list(input_data.shape)
#         # shape_ = delete(shape_input, axis)
#         # 连接列表
#         shape_output = shape_input[:axis] + \
#             list(indices.shape) + shape_input[axis + 1:]
#         data_output = data.reshape(shape_output)
#         return data_output
#     else:
#         data_output = []
#         for data,ind in zip(input_data, indices):
#             r = gather_pytorch(data, ind, batch_dims=batch_dims-1)
#             data_output.append(r)
#         return torch.stack(data_output)

# def gather_pytorch2(input_data, indices=None, batch_dims=0, axis=0):
#     input_data = torch.tensor(input_data)
#     indices = torch.tensor(indices)
#     if axis == None:
#         axis =0 
#     if batch_dims == 0:
#         if axis < 0:
#             axis = len(input_data.shape) + axis
#         data = torch.index_select(input_data, axis, indices.flatten())
#         shape_input = list(input_data.shape)
#         # shape_ = delete(shape_input, axis)
#         # 连接列表
#         shape_output = shape_input[:axis] + \
#             list(indices.shape)[batch_dims:] + shape_input[axis + 1:]
#         data_output = data.reshape(shape_output)
#         return data_output
#     else:
#         data_output = []
#         for data,ind in zip(input_data, indices):
#             r = gather_pytorch2(data, ind, batch_dims=batch_dims-1)
#             data_output.append(r)
#         return torch.stack(data_output)

def gather_nd_pytorch(params, indices):
    '''
    ND_example
    params: tensor shaped [n(1), ..., n(d)] --> d-dimensional tensor
    indices: tensor shaped [m(1), ..., m(i-1), m(i)] --> multidimensional list of i-dimensional indices, m(i) <= d

    returns: tensor shaped [m(1), ..., m(i-1), n(m(i)+1), ..., n(d)] m(i) < d
             tensor shaped [m(1), ..., m(i-1)] m(i) = d
    '''
    indices_shape = indices.shape
    flattened_indices = indices.view(-1, indices.shape[-1])
    processed_tensors = []
    for coordinates in flattened_indices:
        sub_tensor = params[(*coordinates,)] 
        processed_tensors.append(sub_tensor)
    output_shape = indices_shape[:-1] + sub_tensor.shape
    output = torch.stack(processed_tensors).reshape(output_shape)


    return output

def get_real_dtype(dtype):
    """
    Returns the real dtype corresponding to a given complex dtype.
    
    Args:
        dtype: A torch dtype (e.g., torch.complex64, torch.complex128).
        
    Returns:
        The real dtype corresponding to the complex dtype.
    """
    if dtype == torch.complex64:
        return torch.float32
    elif dtype == torch.complex128:
        return torch.float64
    elif dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
        return dtype
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def scatter_nd_add_pytorch(tensor, indices, updates):
    tensor[tuple(indices.t())] += updates
    return tensor
    
def ensure_shape(tensor: torch.Tensor, expected_shape: tuple):
    """
    Ensure the input tensor can broadcast to the specified shape.

    Args:
        tensor (torch.Tensor): The input tensor.
        expected_shape (tuple): The expected shape. Dimensions set to `-1` are ignored in the check.

    Returns:
        torch.Tensor: The input tensor if the shape matches.

    Raises:
        ValueError: If the shape does not match.
    """
    if len(tensor.shape) != len(expected_shape):
        raise ValueError(f"Expected shape with {len(expected_shape)} dimensions, but got {len(tensor.shape)}.")

    for actual, expected in zip(tensor.shape, expected_shape):
        if expected != -1 and actual != expected:
            raise ValueError(f"Expected dimension {expected}, but got {actual}.")
    return tensor

def assert_type(tensor, expected_type):
    assert tensor.dtype == expected_type, f"Expected type {expected_type}, but got {tensor.dtype}"

def arguments_check(params, indices, axis, batch_dims):
    if not (isinstance(params, torch.Tensor) and isinstance(indices, (int, torch.Tensor, np.ndarray)) and isinstance(axis, int) and isinstance(batch_dims, int)):
        raise TypeError(
            f'my_gather() received an invalid combination of arguments - got {(type(params).__name__, type(indices).__name__, type(axis).__name__, type(batch_dims).__name__, )}, but expected one of:\n\
            *(Tensor params, int indices, int axis, int batch_dims)\n\
            *(Tensor params, Tensor indices, int axis, int batch_dims).'
        )
    if not -params.dim() <= axis <= params.dim()-1:
        raise ValueError(
            f'Expected axis in the range [{-params.dim()}, {params.dim()-1}], but got {axis}.'
        )
    if isinstance(indices, int):
        return
    if isinstance(indices, np.ndarray):
        indices = torch.tensor(indices)
    if not -indices.dim() <= batch_dims <= indices.dim():
        raise ValueError(
            f'Expected batch_dims in the range [{-indices.dim()}, {indices.dim()-1}], but got {batch_dims}.'
        )
    axis = axis if axis >= 0 else params.dim()+axis
    batch_dims = batch_dims if batch_dims >= 0 else indices.dim()+batch_dims
    if not batch_dims <= axis:
        raise ValueError(
            f'batch_dims ({batch_dims}) must be less than or equal to axis ({axis}).'
        )
    for index in range(batch_dims):
        if params.shape[index] != indices.shape[index]:
            raise ValueError(
                f'params.shape[{index}]: {params.shape[index]} should be equal to indices.shape[{index}]: {indices.shape[index]}.'
            )

def gather_pytorch(params: torch.Tensor, indices: int | torch.Tensor | np.ndarray, axis: int = 0, batch_dims: int = 0):
    if axis == 0:
        batch_dims = 0
    arguments_check(params, indices, axis, batch_dims)
    axis = axis if axis >= 0 else params.dim()+axis
    if isinstance(indices, int):
        return params.select(dim=axis, index=indices)
    else:
        if isinstance(indices, np.ndarray):
            indices = torch.tensor(indices)
        batch_dims = batch_dims if batch_dims >= 0 else indices.dim()+batch_dims
        output_shape = params.shape[:axis] + indices.shape[batch_dims:] + params.shape[axis+1:]
        indices = indices.to(params.device)
        output = params.index_select(axis, indices.reshape((-1,))).unflatten(axis, indices.shape)
        for index in range(batch_dims):
            output = output.diagonal(dim1=0, dim2=axis-index)
        return output.permute([index+len(output_shape)-batch_dims for index in range(batch_dims)]+[index for index in range(len(output_shape)-batch_dims)])

def argsort_ascending_pytorch(mask: torch.Tensor):
    """
    Args:
        mask: A boolean tensor of shape [..., N]
    Returns:
        data_ind: An integer tensor of shape [..., N] containing the indices that would sort the last dimension of `mask`.
    """
    flattened_tensor = mask.view(-1, mask.shape[-1])
    data_ind = torch.stack([
        torch.tensor(sorted(range(flattened_tensor.shape[-1]), key=lambda x: (flattened_tensor[i, x], x)))
        for i in range(flattened_tensor.shape[0])
    ]).reshape(mask.shape)
    return data_ind

# class RaggedTensor:
#     """
#     A PyTorch implementation of a RaggedTensor-like structure.

#     Attributes:
#         flat_values (torch.Tensor): The flat values of the ragged tensor.
#         row_splits (torch.Tensor): The row splits defining the ragged structure.
#     """

#     def __init__(self, flat_values: torch.Tensor, row_splits: torch.Tensor = None):
#         """
#         Initializes the RaggedTensor.

#         Args:
#             flat_values (torch.Tensor): The flat values of the ragged tensor.
#             row_splits (torch.Tensor): The row splits defining the ragged structure.
#         """
        

#         assert row_splits.ndim == 1, "row_splits must be a 1D tensor."
#         assert row_splits[0] == 0, "row_splits must start with 0."
#         assert torch.all(row_splits[1:] >= row_splits[:-1]), "row_splits must be non-decreasing."
#         assert row_splits[-1] == flat_values.size(0), "Last value of row_splits must equal flat_values size."

#         self.flat_values = flat_values
#         self.row_splits = row_splits

#     @classmethod
#     def from_row_splits(cls, values: torch.Tensor, row_splits: torch.Tensor):
#         """
#         Creates a RaggedTensor from flat values and row splits.

#         Args:
#             values (torch.Tensor): Flat values of the tensor.
#             row_splits (torch.Tensor): Row splits defining the ragged structure.

#         Returns:
#             RaggedTensor: A new instance of RaggedTensor.
#         """
#         return cls(flat_values=values, row_splits=row_splits)

#     def with_flat_values(self, new_flat_values: torch.Tensor):
#         """
#         Creates a new RaggedTensor with the same row_splits but new flat_values.

#         Args:
#             new_flat_values (torch.Tensor): New flat values for the ragged tensor.

#         Returns:
#             RaggedTensor: A new instance of RaggedTensor with updated flat values.
#         """
#         # Ensure the new flat_values size matches the original size
#         assert new_flat_values.size(0) == self.flat_values.size(0), \
#             "new_flat_values must have the same size as the original flat_values."

#         return RaggedTensor(flat_values=new_flat_values, row_splits=self.row_splits)

#     def map_flat_values(self, func: Callable, *args, **kwargs) -> "RaggedTensor":
#         """
#         Applies a function to transform the flat values, or generates new flat values
#         entirely if self.flat_values are not needed.

#         Args:
#             func (Callable): A function to generate new flat values. It can ignore self.flat_values.
#             *args: Positional arguments to pass to the function.
#             **kwargs: Keyword arguments to pass to the function.

#         Returns:
#             RaggedTensor: A new RaggedTensor with transformed or replaced flat values.
#         """
#         # Call the function with or without self.flat_values
#         new_flat_values = func(self.flat_values, *args, **kwargs) if self.flat_values is not None \
#                         else func(*args, **kwargs)
#         return RaggedTensor(new_flat_values, self.row_splits)

#     def to_list(self) -> List[List]:
#         """
#         Converts the ragged tensor to a nested Python list.

#         Returns:
#             List[List]: A nested list representation of the ragged tensor.
#         """
#         result = []
#         for start, end in zip(self.row_splits[:-1], self.row_splits[1:]):
#             result.append(torch.tensor(self.flat_values[start:end].tolist()))
#         return result

#     @classmethod
#     def from_nested_list(cls, nested_list: list, dtype=torch.float32, device=None):
#         flat_values = [item for sublist in nested_list for item in sublist]
#         row_splits = [0]
#         for sublist in nested_list:
#             row_splits.append(row_splits[-1] + len(sublist))

#         flat = torch.tensor(flat_values, dtype=dtype, device=device)
#         splits = torch.tensor(row_splits, dtype=torch.int64, device=device)
#         return cls(flat, splits)

#     def expand_dims(self, dim=-1):
#         """
#         Expands the dimensions of the flat_values tensor.

#         Args:
#             axis (int): The dimension index where the new axis is added.

#         Returns:
#             RaggedTensor: A new RaggedTensor with expanded flat_values.
#         """
#         new_flat_values = self.flat_values.unsqueeze(dim)  # Add dimension
#         return RaggedTensor(flat_values=new_flat_values, row_splits=self.row_splits)

#     def reduce_prod(self, dim = 1, keepdim = False):
#         # Initialize an empty list to store the row-wise products
#         row_products = []

#         # Iterate through the row_splits to compute row products
#         for start, end in zip(self.row_splits[:-1], self.row_splits[1:]):
#             row = self.flat_values[start:end]  # Extract the row
#             if row.numel() > 0:
#                 row_products.append(row.prod(dim-1, keepdim))  # Compute the product of the row
#             else:
#                 row_products.append(torch.tensor(1.0))  # Handle empty rows (product is 1)

#         # Convert the list of row products to a tensor
#         return torch.stack(row_products)
        

#     def reduce_min(self, dim = 1, keepdim = False):
#         # Initialize an empty list to store the row-wise mins
#         row_min = []

#         # Iterate through the row_splits to compute row mins
#         for start, end in zip(self.row_splits[:-1], self.row_splits[1:]):
#             row = self.flat_values[start:end]  # Extract the row
#             if row.numel() > 0:
#                 row_min.append(row.min(dim-1, keepdim).values)  # Compute the min of the row
#             else:
#                 row_min.append(torch.tensor(1.0))  # Handle empty rows (min is 1)
#         return torch.stack(row_min)

#     def reduce_sum(self, dim = 1, keepdim = False):
#         row_sum = []

#         # Iterate through the row_splits to compute row sums
#         for start, end in zip(self.row_splits[:-1], self.row_splits[1:]):
#             row = self.flat_values[start:end]  # Extract the row
#             if row.numel() > 0:
#                 row_sum.append(row.sum(dim-1, keepdim))  # Compute the sum of the row
#             else:
#                 row_sum.append(torch.tensor(1.0))  # Handle empty rows (sum is 1)
#         # first_tensor_shape = row_sum[0].shape
#         # if all(tensor.shape == first_tensor_shape for tensor in row_sum):
#         return torch.stack(row_sum)
        


#     def value_rowids(self):
#         row_ids = torch.cat([torch.full((len(row),), i, dtype=torch.long) for i, row in enumerate(self.to_list())])
#         # values = torch.cat(self.to_list())
#         return row_ids
    
#     def sign(self):
#         sign = torch.sign(self.flat_values)
#         return RaggedTensor(flat_values=sign, row_splits=self.row_splits)
    
#     def __repr__(self):
#         return f"RaggedTensor({self.to_list()})"

class RaggedTensor:
    def __init__(self, flat_values: torch.Tensor, row_splits: torch.Tensor):
        assert row_splits.ndim == 1
        assert row_splits[0] == 0
        assert torch.all(row_splits[1:] >= row_splits[:-1])
        assert row_splits[-1] == flat_values.size(0)

        self.flat_values = flat_values
        self.row_splits = row_splits

    def to(self, device):
        """Move both flat_values and row_splits to a given device."""
        self.flat_values = self.flat_values.to(device)
        self.row_splits = self.row_splits.to(device)
        return self

    @classmethod
    def from_row_splits(cls, values: torch.Tensor, row_splits: torch.Tensor):
        """
        Creates a RaggedTensor from flat values and row splits.

        Args:
            values (torch.Tensor): Flat values of the tensor.
            row_splits (torch.Tensor): Row splits defining the ragged structure.

        Returns:
            RaggedTensor: A new instance of RaggedTensor.
        """
        return cls(flat_values=values, row_splits=row_splits)

    def with_flat_values(self, new_flat_values: torch.Tensor):
        """
        Creates a new RaggedTensor with the same row_splits but new flat_values.

        Args:
            new_flat_values (torch.Tensor): New flat values for the ragged tensor.

        Returns:
            RaggedTensor: A new instance of RaggedTensor with updated flat values.
        """
        # Ensure the new flat_values size matches the original size
        assert new_flat_values.size(0) == self.flat_values.size(0), \
            "new_flat_values must have the same size as the original flat_values."

        return RaggedTensor(flat_values=new_flat_values, row_splits=self.row_splits)

    def map_flat_values(self, func: Callable, *args, **kwargs) -> "RaggedTensor":
        """
        Applies a function to transform the flat values, or generates new flat values
        entirely if self.flat_values are not needed.

        Args:
            func (Callable): A function to generate new flat values. It can ignore self.flat_values.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            RaggedTensor: A new RaggedTensor with transformed or replaced flat values.
        """
        # Call the function with or without self.flat_values
        new_flat_values = func(self.flat_values, *args, **kwargs) if self.flat_values is not None \
                        else func(*args, **kwargs)
        return RaggedTensor(new_flat_values, self.row_splits)

    def to_list(self) -> List[List]:
        """
        Converts the ragged tensor to a nested Python list.

        Returns:
            List[List]: A nested list representation of the ragged tensor.
        """
        result = []
        for start, end in zip(self.row_splits[:-1], self.row_splits[1:]):
            result.append(torch.tensor(self.flat_values[start:end].tolist()))
        return result

    @classmethod
    def from_nested_list(cls, nested_list: list, dtype=torch.float32, device=None):
        flat_values = [item for sublist in nested_list for item in sublist]
        row_splits = [0]
        for sublist in nested_list:
            row_splits.append(row_splits[-1] + len(sublist))

        flat = torch.tensor(flat_values, dtype=dtype, device=device)
        splits = torch.tensor(row_splits, dtype=torch.int64, device=device)
        return cls(flat, splits)

    def expand_dims(self, dim=-1):
        """
        Expands the dimensions of the flat_values tensor.

        Args:
            axis (int): The dimension index where the new axis is added.

        Returns:
            RaggedTensor: A new RaggedTensor with expanded flat_values.
        """
        new_flat_values = self.flat_values.unsqueeze(dim)  # Add dimension
        return RaggedTensor(flat_values=new_flat_values, row_splits=self.row_splits)



    def reduce_prod(self, keepdim=False):
        """Fully GPU-accelerated product per row."""
        device = self.flat_values.device
        num_rows = len(self.row_splits) - 1
        feature_dim = self.flat_values.size(1)
        result = torch.ones((num_rows, feature_dim), device=device, dtype=self.flat_values.dtype)
        for i in range(num_rows):
            start, end = self.row_splits[i].item(), self.row_splits[i + 1].item()
            if start < end:
                result[i] = self.flat_values[start:end].prod(dim=0)
        if keepdim:
            result = result.unsqueeze(-1)
        return result
    # def reduce_prod(self, dim = 1, keepdim = False):
    #     # Initialize an empty list to store the row-wise products
    #     row_products = []

    #     # Iterate through the row_splits to compute row products
    #     for start, end in zip(self.row_splits[:-1], self.row_splits[1:]):
    #         row = self.flat_values[start:end]  # Extract the row
    #         if row.numel() > 0:
    #             row_products.append(row.prod(dim-1, keepdim))  # Compute the product of the row
    #         else:
    #             row_products.append(torch.tensor(1.0))  # Handle empty rows (product is 1)

        # Convert the list of row products to a tensor
        return torch.stack(row_products)
    def reduce_min(self, dim = 1, keepdim = False):
        # Initialize an empty list to store the row-wise mins
        device = self.flat_values.device
        num_rows = len(self.row_splits) - 1
        feature_dim = self.flat_values.size(1)
        result = torch.ones((num_rows, feature_dim), device=device, dtype=self.flat_values.dtype)
        for i in range(num_rows):
            start, end = self.row_splits[i].item(), self.row_splits[i + 1].item()
            if start < end:
                result[i] = self.flat_values[start:end].min()
        if keepdim:
            result = result.unsqueeze(-1)
        return result
    # def reduce_min(self, dim = 1, keepdim = False):
    #     # Initialize an empty list to store the row-wise mins
    #     row_min = []

    #     # Iterate through the row_splits to compute row mins
    #     for start, end in zip(self.row_splits[:-1], self.row_splits[1:]):
    #         row = self.flat_values[start:end]  # Extract the row
    #         if row.numel() > 0:
    #             row_min.append(row.min(dim-1, keepdim).values)  # Compute the min of the row
    #         else:
    #             row_min.append(torch.tensor(1.0))  # Handle empty rows (min is 1)
    def reduce_sum(self, keepdim=False):
        """Fully GPU-accelerated version of reduce_sum."""
        device = self.flat_values.device
        row_ids = torch.repeat_interleave(
            torch.arange(len(self.row_splits) - 1, device=device),
            self.row_splits[1:] - self.row_splits[:-1]
        )
        # result may not be 1D tensor if flat_values is not 1D
        num_rows = len(self.row_splits) - 1
        feature_dim = self.flat_values.size(1)
        result = torch.zeros((num_rows, feature_dim), device=device, dtype=self.flat_values.dtype)
        result.index_add_(0, row_ids, self.flat_values)
        if keepdim:
            result = result.unsqueeze(-1)
        return result
    # def reduce_sum(self, dim = 1, keepdim = False):
    #     row_sum = []

    #     # Iterate through the row_splits to compute row sums
    #     for start, end in zip(self.row_splits[:-1], self.row_splits[1:]):
    #         row = self.flat_values[start:end]  # Extract the row
    #         if row.numel() > 0:
    #             row_sum.append(row.sum(dim-1, keepdim))  # Compute the sum of the row
    #         else:
    #             row_sum.append(torch.tensor(1.0))  # Handle empty rows (sum is 1)
    #     # first_tensor_shape = row_sum[0].shape
    #     # if all(tensor.shape == first_tensor_shape for tensor in row_sum):
    #     return torch.stack(row_sum)

    def value_rowids(self):
        """GPU-friendly row index mapping."""
        device = self.flat_values.device
        row_ids = torch.repeat_interleave(
            torch.arange(len(self.row_splits) - 1, device=device),
            self.row_splits[1:] - self.row_splits[:-1]
        )
        return row_ids
    # def value_rowids(self):
    #     row_ids = torch.cat([torch.full((len(row),), i, dtype=torch.long) for i, row in enumerate(self.to_list())])
    #     # values = torch.cat(self.to_list())
    #     return row_ids
    
    def sign(self):
        return RaggedTensor(torch.sign(self.flat_values), self.row_splits)

    def __repr__(self):
        return f"RaggedTensor(flat_values={self.flat_values}, row_splits={self.row_splits})"
