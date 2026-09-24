"""
Generic operations for torch_nlm and 3D Non-Local Means implementation.
"""

__author__ = "José Guilherme de Almeida"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "José Guilherme de Almeida"
__email__ = "jose.almeida@research.fchampalimaud.org"

# Set PyTorch memory configuration before importing torch
import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import torch.nn.functional as F
import einops
from itertools import product 
from tqdm import tqdm
from typing import Tuple, List
from skimage.restoration import estimate_sigma


# ============================================================================
# Base Operations for torch_nlm
# ============================================================================

def get_gaussian_kernel(kernel_size:int=5, sigma:float=1.,ndim:int=2)->np.ndarray:
    """Creates gaussian kernel with side length kernel_size and a standard 
    deviation of sigma.

    Based on: https://stackoverflow.com/a/43346070

    Args:
        kernel_size (int, optional): size of kernel. Defaults to 5.
        sigma (float, optional): sigma for the normal distribution. Defaults 
            to 1.
        ndim (int, optional): number of dimensions in output kernel.

    Returns:
        np.ndarray: Gaussian filter kernel.
    """
    ax = torch.linspace(-(kernel_size - 1) / 2., 
                        (kernel_size - 1) / 2.,
                        kernel_size)
    gauss = torch.exp(-0.5 * np.square(ax) / np.square(sigma))
    if ndim == 1:
        kernel = gauss
    elif ndim == 2:
        kernel = torch.outer(gauss, gauss)
    elif ndim == 3:
        kernel = gauss[None,None,:] * gauss[None,:,None] * gauss[:,None,None]
    return kernel / torch.sum(kernel)

def unsqueeze_tensor_at_dim(X:torch.Tensor,ndim:int,dim:int=0)->torch.Tensor:
    """
    Adds dimensions as necessary.

    Args:
        X (torch.Tensor): tensor.
        ndim (int): number of output dimensions.
        dim (int): dimension which will be unsqueezed. Defaults to 0.

    Returns:
        torch.Tensor: unqueezed tensor.
    """
    sh = len(X.shape)
    diff = ndim - sh
    if diff > 0:
        for _ in range(diff):
            X = X.unsqueeze(0)
    return X

def make_neighbours_kernel(kernel_size:int=3,ndim:int=2)->torch.Tensor:
    """
    Make convolutional kernel that extracts neighbours within a kernel_size
        neighbourhood (each filter is 1 for the corresponding neighbour and
        0 otherwise). 

    Args:
        kernel_size (int, optional): size of the neighbourhood. Defaults to 3.
        ndim (int, optional): number of dimensions. Only 2 or 3 possible. 
            Defaults to 2.

    Returns:
        torch.Tensor: convolutional kernel for neighbourhood extraction.
    """
    K = kernel_size ** ndim
    filter = torch.zeros([K,1,*[kernel_size for _ in range(ndim)]])
    generators = [range(kernel_size) for _ in range(ndim)]
    for i,coord in enumerate(product(*generators)):
        if ndim == 2:
            filter[i,:,coord[0],coord[1]] = 1
        elif ndim == 3:
            filter[i,:,coord[0],coord[1],coord[2]] = 1
    return filter

def get_neighbours(X:torch.Tensor,
                   kernel_size:int=3,
                   ndim:int=2)->torch.Tensor:
    """
    Retrieves neighbours in an image and stores them in the channel dimension.
    Expects the input to be padded.

    Args:
        X (torch.Tensor): 4-D or 5-D (batched) tensor.
        kernel_size (int, optional): size of the neighbourhood. Defaults to 3.
        ndim (int, optional): number of dimensions. Only 2 or 3 possible. 
            Defaults to 2.

    Returns:
        torch.Tensor: Gaussian filter-normalised X.
    """
    filter = make_neighbours_kernel(kernel_size,ndim).to(X)
    if ndim == 2:
        X = F.conv2d(X,filter,padding=0)
    elif ndim == 3:
        X = F.conv3d(X,filter,padding=0)
    else:
        raise NotImplementedError("ndim must be 2 or 3.")
    return X

def apply_gaussian_filter(X:torch.Tensor,
                          kernel_size:int=3,
                          ndim:int=2,
                          sigma:float=1.0)->torch.Tensor:
    """
    Simple function to apply Gaussian filter.

    Args:
        X (torch.Tensor): input tensor.
        kernel_size (int, optional): size of the neighbourhood. Defaults to 3.
        ndim (int, optional): number of dimensions. Only 2 or 3 possible. 
            Defaults to 2.
        sigma (float, optional): standard deviation for the filter. Defaults to
            1.0.

    Returns:
        torch.Tensor: mean filter-normalised X.
    """
    X = unsqueeze_tensor_at_dim(X,ndim=ndim+2)
    n_channels = X.shape[1]
    gaussian_kernel = get_gaussian_kernel(kernel_size,sigma,ndim)
    gaussian_kernel = unsqueeze_tensor_at_dim(gaussian_kernel,ndim=ndim)
    gaussian_kernel = torch.cat([gaussian_kernel for _ in range(n_channels)],0)
    gaussian_kernel = torch.cat([gaussian_kernel for _ in range(n_channels)],1)
    if ndim == 2:
        X = F.conv2d(X,gaussian_kernel,bias=0,padding=kernel_size//2)
    elif ndim == 3:
        X = F.conv3d(X,gaussian_kernel,bias=0,padding=kernel_size//2)
    else:
        raise NotImplementedError("ndim must be 2 or 3.") 
    return X

def apply_mean_filter(X:torch.Tensor,
                      kernel_size:int=3,
                      ndim:int=2):
    """
    Simple function to apply mean filter.

    Args:
        X (torch.Tensor): input tensor.
        kernel_size (int, optional): size of the neighbourhood. Defaults to 3.
        ndim (int, optional): number of dimensions. Only 2 or 3 possible. 
            Defaults to 2.
        sigma (float, optional): standard deviation for the filter. Defaults to
            1.0.

    Returns:
        torch.Tensor: _description_
    """
    filter = torch.ones([kernel_size for _ in range(ndim)]).to(X)
    filter = filter / filter.sum()
    filter = unsqueeze_tensor_at_dim(filter,ndim+2)
    pad = kernel_size // 2
    padding = tuple([pad for _ in range(ndim * 2)])
    X = F.pad(X,padding,mode="reflect")
    if ndim == 2:
        X = F.conv2d(X,filter,padding=0)
    if ndim == 3:
        X = F.conv3d(X,filter,padding=0)
    return X

def array_chunk(arr:np.ndarray,chunk_size:int):
    for i in range(0,arr.shape[0],chunk_size):
        yield arr[i:i+chunk_size]

def non_local_means_loop_index(X_smooth:torch.Tensor,
                               X_original:torch.Tensor=None,
                               kernel_size:int=3,
                               ndim:int=2,
                               sub_filter_size:int=1,
                               std:float=1.0,
                               patch_kernel:int=1,
                               sigma_est:float=0.0,
                               debug:bool=False)->torch.Tensor:
    """
    Calculates non-local means using a for loop to select at each iteration a
    different set of neighbours. This avoids storing at any given stage a large 
    number of neighbours, making the calculation of larger neighbourhoods 
    possible without running the risk of OOM. Here, neigbhours are selected 
    using simple indexing.

    Args:
        X_smooth (torch.Tensor): smoothed input tensor used for distance calculation (patch matching).
        X_original (torch.Tensor, optional): original unsmoothed tensor for output averaging. 
            If None, uses X_smooth for both. Defaults to None.
        kernel_size (int, optional): size of neighbourhood. Defaults to 3.
        ndim (int, optional): number of dimensions. Must be 2 or 3. Defaults to
            2.
        sub_filter_size (int, optional): approximate size of neighbourhood set
            at each iteration. Defaults to 1 (regular non local means).
        std (float, optional): standard deviation for weights. Defaults to 1.0.
        debug (bool, optional): enable debugging output. Defaults to False.

    Returns:
        torch.Tensor: non local mean-normalised input.
    """
    if X_original is None:
        X_original = X_smooth
    
    if debug:
        print(f"\n[DEBUG] non_local_means_loop_index called")
        print(f"[DEBUG] X_smooth shape: {X_smooth.shape}, X_original shape: {X_original.shape}, kernel_size: {kernel_size}, std: {std}")
    
    def cat_idxs(idxs:List[torch.Tensor]):
        idxs = torch.cat(idxs,1)
        return idxs

    def preprocess_idxs(idxs: List[torch.Tensor],
                        original_idxs: List[torch.Tensor],
                        sizes: List[int]):
        d = torch.abs(original_idxs - idxs)
        idxs = torch.where(idxs < 0, d - original_idxs, idxs)
        idxs = torch.where(idxs > sizes - 1, original_idxs - d, idxs)
        return idxs

    def weight_operation(X_ref, neighbours, std_2):
        # scikit-image formula: exp(-(distance^2) / (2 * h^2))
        diff2 = torch.square(X_ref - neighbours)
        if patch_kernel > 1:
            pad = patch_kernel // 2
            diff2 = F.avg_pool3d(diff2, patch_kernel, stride=1, padding=pad) * (patch_kernel ** ndim)
        # Apply noise variance compensation (scikit-image style)
        if sigma_est > 0:
            noise_term = 2.0 * (sigma_est ** 2) * (patch_kernel ** ndim)
            diff2 = torch.clamp(diff2 - noise_term, min=0.0)
        return torch.exp(-diff2 / (std_2 + 1e-8))

    def calculate_weights_2d(X_smooth:torch.Tensor,
                             X_original:torch.Tensor,
                             idxs:List[torch.Tensor],
                             std_2:torch.Tensor)->Tuple[torch.Tensor,
                                                        torch.Tensor]:
        n = len(idxs)
        idxs = cat_idxs(idxs)
        # Extract smoothed neighbors for distance calculation
        neighbours_smooth = X_smooth[:,:,idxs[0],idxs[1]].reshape(
            1,n,X_smooth.shape[2],X_smooth.shape[3])
        # Extract original neighbors for output averaging
        neighbours_original = X_original[:,:,idxs[0],idxs[1]].reshape(
            1,n,X_original.shape[2],X_original.shape[3])
        X_ref = X_original if X_original is not None else X_smooth
        # Compute weights from smoothed data
        weights = weight_operation(X_ref, neighbours_original, std_2)
        # Return weights and original neighbors for averaging
        return weights, neighbours_original

    def calculate_weights_3d(X_smooth:torch.Tensor,
                             X_original:torch.Tensor,
                             idxs:List[torch.Tensor],
                             std_2:torch.Tensor)->Tuple[torch.Tensor,
                                                        torch.Tensor]:
        n = len(idxs)
        idxs = cat_idxs(idxs)
        # Extract smoothed neighbors for distance calculation
        neighbours_smooth = X_smooth[:,:,idxs[0],idxs[1],idxs[2]].reshape(
            1,n,X_smooth.shape[2],X_smooth.shape[3],X_smooth.shape[4])
        # Extract original neighbors for output averaging
        neighbours_original = X_original[:,:,idxs[0],idxs[1],idxs[2]].reshape(
            1,n,X_original.shape[2],X_original.shape[3],X_original.shape[4])
        # Compute weights from smoothed data
        X_ref = X_original if X_original is not None else X_smooth
        weights = weight_operation(X_ref, neighbours_original, std_2)
        # Return weights and original neighbors for averaging
        return weights, neighbours_original

    weights_sum = torch.zeros_like(X_original)
    output = torch.zeros_like(X_original)
    k2 = kernel_size // 2
    std_2 = torch.as_tensor(std**2).to(X_smooth)
    counter = 0
    sample_weights = []  # For debugging
    sample_distances = []
    
    if ndim == 2:
        _,_,H,W = X_smooth.shape
        coords = torch.stack(torch.where(X_smooth[0,0] == X_smooth[0,0]))
        tmp_idxs = torch.zeros_like(coords)
        size = torch.as_tensor([H,W]).reshape(ndim,1).to(coords)
        all_idxs = []
        range_h = torch.arange(-k2,k2+1)
        range_w = torch.arange(-k2,k2+1)
        for i,j in product(range_h,range_w):
            counter += 1
            tmp_idxs[0][:] = coords[0] + i
            tmp_idxs[1][:] = coords[1] + j
            all_idxs.append(preprocess_idxs(tmp_idxs,coords,size))
            if counter >= sub_filter_size:
                weights,neighbours = calculate_weights_2d(X_smooth,X_original,all_idxs,std_2)
                if debug and len(sample_weights) < 3:
                    sample_weights.append(weights.clone().cpu())
                    sample_distances.append(torch.sqrt(torch.square(X_smooth.cpu() - X_smooth[:,:,all_idxs[0][0],all_idxs[0][1]].cpu())))
                weights_sum += torch.sum(weights,1)
                output += weights.multiply(neighbours).sum(1)
                all_idxs = []
                counter = 0
        if counter > 0:
            weights,neighbours = calculate_weights_2d(X_smooth,X_original,all_idxs,std_2)
            if debug and len(sample_weights) < 3:
                sample_weights.append(weights.clone().cpu())
                sample_distances.append(torch.sqrt(torch.square(X_smooth.cpu() - X_smooth[:,:,all_idxs[0][0],all_idxs[0][1]].cpu())))
            weights_sum += torch.sum(weights,1)
            output += weights.multiply(neighbours).sum(1)

    if ndim == 3:
        _,_,H,W,D = X_smooth.shape
        coords = torch.stack(torch.where(X_smooth[0,0] == X_smooth[0,0]))
        tmp_idxs = torch.zeros_like(coords)
        size = torch.as_tensor([H,W,D]).reshape(ndim,1).to(coords)
        all_idxs = []
        range_h = torch.arange(-k2,k2+1)
        range_w = torch.arange(-k2,k2+1)
        range_d = torch.arange(-k2,k2+1)
        for i,j,k in product(range_h,range_w,range_d):
            counter += 1
            tmp_idxs[0][:] = coords[0] + i
            tmp_idxs[1][:] = coords[1] + j
            tmp_idxs[2][:] = coords[2] + k
            all_idxs.append(preprocess_idxs(tmp_idxs,coords,size))
            if counter >= sub_filter_size:
                weights,neighbours = calculate_weights_3d(X_smooth,X_original,all_idxs,std_2)
                if debug and len(sample_weights) < 3:
                    sample_weights.append(weights.clone().cpu())
                    sample_distances.append(torch.sqrt(torch.square(X_smooth.cpu() - X_smooth[:,:,all_idxs[0][0],all_idxs[0][1],all_idxs[0][2]].cpu())))
                weights_sum += torch.sum(weights,1)
                output += weights.multiply(neighbours).sum(1)
                all_idxs = []
                counter = 0
        if counter > 0:
            weights,neighbours = calculate_weights_3d(X_smooth,X_original,all_idxs,std_2)
            if debug and len(sample_weights) < 3:
                sample_weights.append(weights.clone().cpu())
                sample_distances.append(torch.sqrt(torch.square(X_smooth.cpu() - X_smooth[:,:,all_idxs[0][0],all_idxs[0][1],all_idxs[0][2]].cpu())))
            weights_sum += torch.sum(weights,1)
            output += weights.multiply(neighbours).sum(1)

    if debug:
        print(f"\n[DEBUG] Weight statistics:")
        for i, w in enumerate(sample_weights):
            print(f"  Sample {i}: mean={w.mean():.6f}, max={w.max():.6f}, min={w.min():.6f}, near_zero={(w < 1e-6).sum()}")
        print(f"\n[DEBUG] Distance statistics:")
        for i, d in enumerate(sample_distances):
            print(f"  Sample {i}: mean={d.mean():.6f}, max={d.max():.6f}, min={d.min():.6f}")
        
        weights_sum_cpu = weights_sum.squeeze().cpu() if weights_sum.numel() > 1 else weights_sum.cpu()
        print(f"\n[DEBUG] Weight sum (denominator):")
        print(f"  Range: [{weights_sum_cpu.min():.6f}, {weights_sum_cpu.max():.6f}]")
        print(f"  Mean: {weights_sum_cpu.mean():.6f}")

    return output / weights_sum

def non_local_means_loop(X:torch.Tensor,
                         kernel_size:int=3,
                         ndim:int=2,
                         sub_filter_size:int=1,
                         std:float=1.0)->torch.Tensor:
    """
    Calculates non-local means using a for loop to select at each iteration a
    different set of neighbours. This avoids storing at any given stage a large 
    number of neighbours, making the calculation of larger neighbourhoods 
    possible without running the risk of OOM. Here, neigbhours are selected 
    using convolutional filters.

    Args:
        X (torch.Tensor): input tensor.
        kernel_size (int, optional): size of neighbourhood. Defaults to 3.
        ndim (int, optional): number of dimensions. Must be 2 or 3. Defaults to
            2.
        sub_filter_size (int, optional): approximate size of neighbourhood set
            at each iteration. Defaults to 1 (regular non local means).
        std (float, optional): standard deviation for weights. Defaults to 1.0.

    Returns:
        torch.Tensor: non local mean-normalised input.
    """
    filter = make_neighbours_kernel(kernel_size,ndim).to(X)
    pad_size = kernel_size // 2
    padding = tuple([pad_size for _ in range(4)])
    weights_sum = torch.zeros_like(X)
    output = torch.zeros_like(X)
    n_filters = filter.shape[0]
    if sub_filter_size > n_filters:
        blocks = [np.arange(n_filters,dtype=int)]
    else:
        blocks = list(
            array_chunk(np.arange(n_filters,dtype=int),sub_filter_size))
    padded_X = F.pad(X,padding)
    std_2 = torch.as_tensor(std**2).to(X)
    with torch.no_grad():
        for block in tqdm(blocks):
            neighbours = F.conv2d(
                padded_X,filter[block],padding=0)
            # scikit-image formula: exp(-(distance^2) / (2 * h^2))
            weights = torch.square(X - neighbours).negative().divide(std_2 / 2.0).exp()
            weights_sum += torch.sum(weights,1)
            output += weights.multiply(neighbours).sum(1)
    output = output / weights_sum
    return output

# ============================================================================
# 3D Non-Local Means Implementation
# ============================================================================

def apply_nonlocal_means_3d(X:torch.Tensor,
                            kernel_size:int=3,
                            std:float=1,
                            kernel_size_mean=3):
    """
    Calculates non-local means for an input X with 3 dimensions.

    Args:
        X (torch.Tensor): input tensor with shape [h,w,d].
        kernel_size (int, optional): size of neighbourhood. Defaults to 3.
        std (float, optional): standard deviation for weights. Defaults to 1.0.
        kernel_size_mean (int, optional): kernel size for the initial mean
        filtering.

    Returns:
        torch.Tensor: non local mean-normalised input.
    """
    ndim = 3
    # include batch and channel dimensions
    X = unsqueeze_tensor_at_dim(X,ndim+2)
    padding = (
        kernel_size // 2, kernel_size // 2, 
        kernel_size // 2, kernel_size // 2,
        kernel_size // 2, kernel_size // 2)
    # Store original for final averaging
    X_original = X.clone()
    # Apply mean filter for patch comparison
    X = apply_mean_filter(X,kernel_size_mean,ndim)
    # retrieve neighbourhood
    neighbours_X = get_neighbours(
        F.pad(X,padding),kernel_size=kernel_size,ndim=ndim)
    # Get original neighbors for output
    neighbours_X_original = get_neighbours(
        F.pad(X_original,padding),kernel_size=kernel_size,ndim=ndim)
    distances = torch.sqrt(torch.square(X - neighbours_X))
    # scikit-image formula: exp(-(distance^2) / (2 * h^2))
    weights = torch.exp(- distances / (2 * std**2))
    weights = weights / weights.sum(1,keepdim=True)
    output = (neighbours_X_original * weights).sum(1).squeeze(0)
    return output

def apply_windowed_nonlocal_means_3d(X:torch.Tensor,
                                     kernel_size:int=3,
                                     std:int=1,
                                     kernel_size_mean=3,
                                     window_size:Tuple[int,int,int]=[128,128,128],
                                     strides:Tuple[int,int,int]=[64,64,64]):
    """
    Calculates non-local means for an input X with 3 dimensions by calculating 
    a local (windowed) non-local means. Leads to artefacts because of this - 
    not really good as an implementation, kept mostly as a curiosity.

    Args:
        X (torch.Tensor): input tensor with shape [h,w,d].
        kernel_size (int, optional): size of neighbourhood. Defaults to 3.
        std (float, optional): standard deviation for weights. Defaults to 1.0.
        kernel_size_mean (int, optional): kernel size for the initial mean
            filtering.
        window_size (Tuple[int,int,int], optional): size of window. Defaults to
            [128,128,128].
        strides (Tuple[int,int,int], optional): size of window. Defaults to
            [64,64,64].

    Returns:
        torch.Tensor: non local mean-normalised input.
    """
    ndim = 3
    output = torch.zeros_like(X)
    denominator = torch.zeros_like(X)
    # include batch and channel dimensions
    X = unsqueeze_tensor_at_dim(X,ndim+2)
    # Store original for final averaging
    X_original = X.clone()
    # Apply mean filter for patch comparison
    X = apply_mean_filter(X,kernel_size_mean,ndim)
    sh = X.shape
    b,c,h,w,d = sh[0],sh[1],sh[2],sh[3],sh[4]
    padding = (
        kernel_size // 2, kernel_size // 2, 
        kernel_size // 2, kernel_size // 2,
        kernel_size // 2, kernel_size // 2)
    X = F.pad(X,padding)
    X_original = F.pad(X_original,padding)
    neighbours_X = get_neighbours(X,kernel_size=kernel_size,ndim=ndim)
    neighbours_X_original = get_neighbours(X_original,kernel_size=kernel_size,ndim=ndim)
    X = X[:,:,
          padding[0]:(X.shape[2]-padding[0]),
          padding[1]:(X.shape[3]-padding[1]),
          padding[2]:(X.shape[3]-padding[2])]
    
    all_ij = list(product(range(0,h,strides[0]),
                          range(0,w,strides[1]),
                          range(0,d,strides[2])))

    for i,j,k in tqdm(all_ij):
        i_1,i_2 = i,i+window_size[0]
        j_1,j_2 = j,j+window_size[1]
        k_1,k_2 = k,k+window_size[2]
        if i_2 > h:
            i_1,i_2 = h - window_size[0], h
        if j_2 > w:
            j_1,j_2 = w - window_size[1], w
        if k_2 > w:
            k_1,k_2 = d - window_size[2], d
        # reshape X to calculate distances from smoothed data
        sub_X = X[:,:,i_1:i_2,j_1:j_2,k_1:k_2]
        sub_neighbours = neighbours_X[:,:,i_1:i_2,j_1:j_2,k_1:k_2]
        # Use original for output
        sub_neighbours_original = neighbours_X_original[:,:,i_1:i_2,j_1:j_2,k_1:k_2]
        reshaped_neighbours_X = einops.rearrange(
            sub_neighbours,"b c h w d -> b (h w d) c")
        # calculate distances from smoothed data
        neighbour_dists = torch.cdist(
            reshaped_neighbours_X,reshaped_neighbours_X)
        # calculate weights from distances using scikit-image formula: exp(-(distance^2) / (2 * h^2))
        weights = torch.exp(-neighbour_dists / (2 * std**2))
        # calculate the new values using original neighbors
        reshaped_neighbours_original = einops.rearrange(
            sub_neighbours_original,"b c h w d -> b (h w d) c")
        weighted_X = (weights @ reshaped_neighbours_original).squeeze(-1)
        output[i_1:i_2,j_1:j_2,k_1:k_2] += einops.rearrange(
            weighted_X,"b (h w d) -> b h w d",
            b=b,h=window_size[0],w=window_size[1],d=window_size[2]).squeeze(0)
        denominator[i_1:i_2,j_1:j_2,k_1:k_2] += einops.rearrange(
            weights.sum(-1),"b (h w d) -> b h w d",
            b=b,h=window_size[0],w=window_size[1],d=window_size[2]).squeeze(0)
    output = output / denominator
    return output

def apply_nonlocal_means_3d_mem_efficient(X:torch.Tensor,
                                          kernel_size:int=3,
                                          std:int=1,
                                          kernel_size_mean=3,
                                          sub_filter_size:int=256,
                                          debug:bool=False):
    """
    Calculates non-local means using a for loop to select at each iteration a
    different set of neighbours. Most of the heavy lifting is performed by
    non_local_means_loop.

    Args:
        X (torch.Tensor): input tensor with shape [h,w,d].
        kernel_size (int, optional): size of neighbourhood. Defaults to 3.
        std (float, optional): standard deviation for weights. Defaults to 1.0.
        kernel_size_mean (int, optional): kernel size for the initial mean
            filtering.
        sub_filter_size (int, optional): approximate size of neighbourhood set
            at each iteration. Defaults to 1 (regular non local means).
        debug (bool, optional): enable debugging output. Defaults to False.

    Returns:
        torch.Tensor: non local mean-normalised input.
    """
    if debug:
        print("\n[DEBUG] Using apply_nonlocal_means_3d_mem_efficient path")
        print(f"[DEBUG] Input shape: {X.shape}")
        if X.is_cuda:
            print(f"[DEBUG] Input device: {X.device}")
            X_cpu = X.cpu()
        else:
            X_cpu = X
        print(f"[DEBUG] Input data range: [{X_cpu.min():.4f}, {X_cpu.max():.4f}]")
        print(f"[DEBUG] Input data mean: {X_cpu.mean():.4f}, std: {X_cpu.std():.4f}")
    
    ndim = 3
    # include batch and channel dimensions
    X_unsqueezed = unsqueeze_tensor_at_dim(X, ndim+2)
    
    # Store ORIGINAL unsmoothed data for final averaging
    X_original = X_unsqueezed.clone()
    
    # Estimate noise sigma from input data (used for noise variance compensation)
    X_np = X.squeeze().cpu().numpy() if X.ndim > 3 else X.cpu().numpy()
    sigma_est = float(estimate_sigma(X_np, channel_axis=None, average_sigmas=True))
    
    # Apply mean filter for patch comparison (patch_size parameter)
    # Apply to original unsqueezed tensor, not the clone
    #X_smooth = apply_mean_filter(X_unsqueezed, kernel_size_mean, ndim)
    
    # retrieve neighbourhood using smoothed data for distance, original for output
    output = non_local_means_loop_index(
        X_original,  # supply raw data for distance
        X_original=X_original,
        kernel_size=kernel_size,
        std=std,
        ndim=ndim,
        sub_filter_size=sub_filter_size,
        patch_kernel=kernel_size_mean,
        sigma_est=sigma_est,
        debug=debug
    ).squeeze(0).squeeze(0)
    
    if debug:
        output_cpu = output.cpu() if output.is_cuda else output
        print(f"[DEBUG] Output data range: [{output_cpu.min():.4f}, {output_cpu.max():.4f}]")
        print(f"[DEBUG] Output data mean: {output_cpu.mean():.4f}, std: {output_cpu.std():.4f}")
    
    return output

# ============================================================================
# GPU Tiled Processing with OOM Fallback (Streaming Neighbours)
# ============================================================================

import math

def _estimate_free_mem_gb():
    """Estimate free GPU memory in gigabytes."""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return free / 1024**3
    return 0.0

def _offsets_3d(kernel_size: int):
    """Generate 3D offsets for NLM neighbourhood."""
    k2 = kernel_size // 2
    return [(i, j, k) for i in range(-k2, k2 + 1)
                    for j in range(-k2, k2 + 1)
                    for k in range(-k2, k2 + 1)]

def _chunk(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

@torch.no_grad()
def _process_tile_streaming(tile_cpu: torch.Tensor,
                            kernel_size: int,
                            std: float,
                            kernel_size_mean: int,
                            target_block_mem_gb: float = 0.5,
                            sigma_est: float = 0.0,
                            debug: bool = False) -> torch.Tensor:
    """
    Memory-efficient NLM on a single tile using streaming neighbours.
    Smooths data for patch matching, then averages original unsmoothed neighbors.
    
    Args:
        sigma_est: Pre-estimated noise sigma. If <= 0, will estimate from tile.
    """
    assert tile_cpu.device.type == 'cpu'
    device = torch.device('cuda')
    H, W, D = tile_cpu.shape
    
    if debug:
        print(f"\n[DEBUG] Tile shape: {H}x{W}x{D}")
        print(f"[DEBUG] Input data range: [{tile_cpu.min():.4f}, {tile_cpu.max():.4f}]")
        print(f"[DEBUG] Input data mean: {tile_cpu.mean():.4f}, std: {tile_cpu.std():.4f}")
    
    # Move tile to GPU
    X = tile_cpu.to(device, non_blocking=True)
    X = unsqueeze_tensor_at_dim(X, ndim=5)  # [1,1,H,W,D]

    ndim = 3
    
    # Store ORIGINAL unsmoothed data for final averaging
    X_original = X.clone()

    # Apply reflect padding to ORIGINAL data
    pad = kernel_size // 2
    padding = (pad, pad, pad, pad, pad, pad)
    Xp_original = F.pad(X_original, padding, mode="reflect")

    # Estimate noise sigma ONLY if not provided (to avoid per-tile overhead)
    if sigma_est <= 0:
        tile_cpu_np = tile_cpu.squeeze().cpu().numpy() if tile_cpu.ndim > 3 else tile_cpu.cpu().numpy()
        sigma_est = float(estimate_sigma(tile_cpu_np, channel_axis=None, average_sigmas=True))
        if debug:
            print(f"[DEBUG] Estimated noise sigma from tile: {sigma_est:.6f}")
    else:
        if debug:
            print(f"[DEBUG] Using provided noise sigma: {sigma_est:.6f}")
    
    if debug:
        print(f"[DEBUG] Kernel size: {kernel_size}, Kernel size mean: {kernel_size_mean}")
        print(f"[DEBUG] std parameter (h): {std}, std²: {std**2}")

    # Accumulators
    num = torch.zeros_like(X, device=device)  # [1,1,H,W,D]
    den = torch.zeros_like(X, device=device)  # [1,1,H,W,D]
    std2 = float(std) ** 2

    # Prepare offsets
    offs = _offsets_3d(kernel_size)
    
    if debug:
        print(f"[DEBUG] Number of offsets: {len(offs)}")
    
    # Decide subK based on memory
    vox = H * W * D
    bytes_per_vox = 4.0
    if target_block_mem_gb <= 0:
        target_block_mem_gb = 0.5
    subK = max(1, int((target_block_mem_gb * (1024**3)) / (2.0 * vox * bytes_per_vox)))
    subK = min(subK, len(offs))

    # Process in blocks of offsets
    weights_stats = []  # For debugging
    distances_stats = []
    
    def process_block(neigh_offsets, block_idx=0):
        nonlocal num, den, weights_stats, distances_stats
        for offset_idx, (di, dj, dk) in enumerate(neigh_offsets):
            xs, xe = di + pad, di + pad + H
            ys, ye = dj + pad, dj + pad + W
            zs, ze = dk + pad, dk + pad + D
            
            # Compute weights from ORIGINAL unsmoothed data (for patch matching)
            neigh_original = Xp_original[:, :, xs:xe, ys:ye, zs:ze]
            diff2 = torch.square(X_original - neigh_original)
            
            # Aggregate over patch kernel size (accumulate squared differences)
            if kernel_size_mean > 1:
                patch_pad = kernel_size_mean // 2
                diff2 = F.avg_pool3d(
                    diff2,
                    kernel_size=kernel_size_mean,
                    stride=1,
                    padding=patch_pad
                ) * (kernel_size_mean ** 3)
            
            # Apply noise variance compensation (scikit-image style)
            noise_term = 2.0 * (sigma_est ** 2) * (kernel_size_mean ** 3)
            dist = torch.clamp(diff2 - noise_term, min=0.0)
            
            # Debug: record distance statistics for first few offsets
            if debug and (block_idx == 0 and offset_idx < 5):
                dist_vals = torch.sqrt(dist).cpu()
                distances_stats.append({
                    'offset': (di, dj, dk),
                    'mean_distance': dist_vals.mean().item(),
                    'max_distance': dist_vals.max().item(),
                    'min_distance': dist_vals.min().item()
                })
            
            w = torch.exp(-dist / (std2 + 1e-8))
            
            # Debug: record weight statistics for first few offsets
            if debug and (block_idx == 0 and offset_idx < 5):
                w_cpu = w.cpu()
                weights_stats.append({
                    'offset': (di, dj, dk),
                    'mean_weight': w_cpu.mean().item(),
                    'max_weight': w_cpu.max().item(),
                    'min_weight': w_cpu.min().item(),
                    'num_near_zero': (w_cpu < 1e-6).sum().item()
                })
            
            # Accumulate weighted average of ORIGINAL unsmoothed neighbors
            num += w * neigh_original
            den += w
            
            del neigh_original, diff2, dist, w

    try:
        for block_idx, block in enumerate(_chunk(offs, subK)):
            process_block(block, block_idx=block_idx)
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"    OOM in block processing, switching to single-offset mode")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            for (di, dj, dk) in offs:
                process_block([(di, dj, dk)])
            torch.cuda.empty_cache()
        else:
            raise

    # Debug output
    if debug:
        print(f"\n[DEBUG] Distance statistics (first 5 offsets):")
        for d in distances_stats:
            print(f"  Offset {d['offset']}: mean={d['mean_distance']:.6f}, "
                  f"min={d['min_distance']:.6f}, max={d['max_distance']:.6f}")
        
        print(f"\n[DEBUG] Weight statistics (first 5 offsets):")
        for w in weights_stats:
            print(f"  Offset {w['offset']}: mean={w['mean_weight']:.6f}, "
                  f"max={w['max_weight']:.6f}, min={w['min_weight']:.6f}, "
                  f"near_zero={w['num_near_zero']}")
        
        den_cpu = den.squeeze(0).squeeze(0).cpu()
        num_cpu = num.squeeze(0).squeeze(0).cpu()
        print(f"\n[DEBUG] Denominator (weight sum) stats:")
        print(f"  Range: [{den_cpu.min():.6f}, {den_cpu.max():.6f}]")
        print(f"  Mean: {den_cpu.mean():.6f}")
        print(f"  Any near-zero: {(den_cpu < 1e-6).sum().item()}")

    out = (num / (den + 1e-8)).squeeze(0).squeeze(0).cpu()
    
    if debug:
        print(f"\n[DEBUG] Output data range: [{out.min():.4f}, {out.max():.4f}]")
        print(f"[DEBUG] Output data mean: {out.mean():.4f}, std: {out.std():.4f}")
        print(f"[DEBUG] Difference (output - input): range=[{(out - tile_cpu).min():.6f}, {(out - tile_cpu).max():.6f}]")
    
    # Cleanup
    del X, X_original, Xp_original, num, den
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return out

@torch.no_grad()
def apply_nonlocal_means_3d_tiled(X: torch.Tensor,
                                  kernel_size: int = 3,
                                  std: float = 1.0,
                                  kernel_size_mean: int = 3,
                                  tile_size: int = 128,
                                  sigma_est: float = 0.0,
                                  debug: bool = False):
    """
    Tiled NLM with true CPU↔GPU streaming and per-offset streaming.
    Passes full context-padded tiles to processing to avoid boundary artifacts.
    
    Args:
        sigma_est: Pre-estimated noise sigma for all tiles. If <= 0, estimates once globally.
    """
    # Normalize input shape
    if X.ndim == 5:
        X = X.squeeze(0).squeeze(0)
    # Ensure CPU master copy to avoid holding big GPU allocation
    if X.is_cuda:
        X = X.cpu()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    H, W, D = X.shape
    out = torch.zeros_like(X, device='cpu')
    
    # Estimate noise sigma ONCE for the entire volume (if not provided)
    if sigma_est <= 0:
        X_np = X.squeeze().cpu().numpy() if X.ndim > 3 else X.cpu().numpy()
        sigma_est = float(estimate_sigma(X_np, channel_axis=None, average_sigmas=True))
        if debug:
            print(f"\n[DEBUG] Estimated global noise sigma: {sigma_est:.6f}")

    # Context padding for neighbourhood and mean filtering
    pad = kernel_size // 2 + kernel_size_mean // 2

    # Build tile ranges
    xr = [(x, min(x + tile_size, H)) for x in range(0, H, tile_size)]
    yr = [(y, min(y + tile_size, W)) for y in range(0, W, tile_size)]
    zr = [(z, min(z + tile_size, D)) for z in range(0, D, tile_size)]

    print(f"\n{'='*70}")
    print("TILED GPU DENOISING (Streaming Neighbours with Context)")
    print(f"{'='*70}")
    print(f"Volume size: {H}x{W}x{D} = {H*W*D/1e9:.2f}B voxels")
    print(f"Tile size: {tile_size}x{tile_size}x{tile_size}")
    print(f"Context padding: {pad}px")
    print(f"Total tiles: {len(xr)*len(yr)*len(zr)}")
    print(f"Strategy: Full context → NLM → extract center (no boundary artifacts)")
    print(f"{'='*70}\n")

    tile_idx = 0
    for (x0, x1) in xr:
        for (y0, y1) in yr:
            for (z0, z1) in zr:
                tile_idx += 1
                
                # Extract full padded tile with context
                xp0, xp1 = max(0, x0 - pad), min(H, x1 + pad)
                yp0, yp1 = max(0, y0 - pad), min(W, y1 + pad)
                zp0, zp1 = max(0, z0 - pad), min(D, z1 + pad)
                
                tile_padded = X[xp0:xp1, yp0:yp1, zp0:zp1].clone()

                # Compute offsets inside padded tile where we want results
                cx0 = x0 - xp0
                cx1 = cx0 + (x1 - x0)
                cy0 = y0 - yp0
                cy1 = cy0 + (y1 - y0)
                cz0 = z0 - zp0
                cz1 = cz0 + (z1 - z0)

                # Estimate a safe per-block mem (use ~25% of free mem)
                free_gb = _estimate_free_mem_gb()
                block_budget = max(0.25, free_gb * 0.25)

                print(f"Tile {tile_idx}/{len(xr)*len(yr)*len(zr)}: "
                      f"[{x0}:{x1}, {y0}:{y1}, {z0}:{z1}] "
                      f"Padded: {tuple(tile_padded.shape)} "
                      f"(GPU free ~{free_gb:.2f} GB)")

                try:
                    # Process FULL padded tile (with context), not just core
                    den_full = _process_tile_streaming(
                        tile_padded,
                        kernel_size=kernel_size,
                        std=std,
                        kernel_size_mean=kernel_size_mean,
                        target_block_mem_gb=block_budget,
                        sigma_est=sigma_est,
                        debug=debug
                    )
                    # Extract only the center region (non-padded output)
                    den_core = den_full[cx0:cx1, cy0:cy1, cz0:cz1]
                    print(f"  ✓ Complete")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Extreme backoff: process with tiny sub-tiles
                        print(f"  ⚠ OOM in streaming; retrying with tiny tiles (64)")
                        small = 64
                        den_core = torch.zeros(x1 - x0, y1 - y0, z1 - z0)
                        for sx in range(0, x1 - x0, small):
                            for sy in range(0, y1 - y0, small):
                                for sz in range(0, z1 - z0, small):
                                    # Extract sub-tile WITH padding from original
                                    sx_pad = max(0, x0 + sx - pad)
                                    sx_end = min(H, x0 + sx + small + pad)
                                    sy_pad = max(0, y0 + sy - pad)
                                    sy_end = min(W, y0 + sy + small + pad)
                                    sz_pad = max(0, z0 + sz - pad)
                                    sz_end = min(D, z0 + sz + small + pad)
                                    
                                    sub_tile = X[sx_pad:sx_end, sy_pad:sy_end, sz_pad:sz_end].clone()
                                    
                                    # Offsets for extraction
                                    sub_cx0 = (x0 + sx) - sx_pad
                                    sub_cx1 = sub_cx0 + min(small, x1 - x0 - sx)
                                    sub_cy0 = (y0 + sy) - sy_pad
                                    sub_cy1 = sub_cy0 + min(small, y1 - y0 - sy)
                                    sub_cz0 = (z0 + sz) - sz_pad
                                    sub_cz1 = sub_cz0 + min(small, z1 - z0 - sz)
                                    
                                    den_sub = _process_tile_streaming(
                                        sub_tile, kernel_size, std, 
                                        kernel_size_mean=kernel_size_mean,
                                        target_block_mem_gb=0.25,
                                        sigma_est=sigma_est,
                                        debug=debug
                                    )
                                    
                                    den_core[sx:sx+den_sub[sub_cx0:sub_cx1, sub_cy0:sub_cy1, sub_cz0:sub_cz1].shape[0],
                                             sy:sy+den_sub[sub_cx0:sub_cx1, sub_cy0:sub_cy1, sub_cz0:sub_cz1].shape[1],
                                             sz:sz+den_sub[sub_cx0:sub_cx1, sub_cy0:sub_cy1, sub_cz0:sub_cz1].shape[2]] = \
                                        den_sub[sub_cx0:sub_cx1, sub_cy0:sub_cy1, sub_cz0:sub_cz1]
                    else:
                        raise

                out[x0:x1, y0:y1, z0:z1] = den_core
                # Free per-tile memory
                del tile_padded, den_core, den_full
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    print(f"\n{'='*70}")
    print("Tiled streaming denoising complete")
    print(f"{'='*70}\n")
    return out


def apply_nonlocal_means_3d_with_fallback(X: torch.Tensor,
                                          kernel_size: int = 3,
                                          std: float = 1.0,
                                          kernel_size_mean: int = 3,
                                          sub_filter_size: int = 256,
                                          debug: bool = False):
    """
    Wrapper that attempts full 3D NLM processing, with automatic fallback
    to streaming-tiled processing if GPU runs out of memory.
    
    This allows processing of large volumes on 16GB GPUs using per-offset
    streaming to avoid the K-channel neighbour explosion.
    """
    
    try:
        print("\n[GPU NLM] Attempting full 3D processing...")
        return apply_nonlocal_means_3d_mem_efficient(
            X,
            kernel_size=kernel_size,
            std=std,
            kernel_size_mean=kernel_size_mean,
            sub_filter_size=sub_filter_size
        )
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n⚠️  OOM in full 3D mode")
            print("Switching to streaming tiled processing...\n")
            
            # Make a CPU copy to avoid depending on freeing caller's GPU tensor
            X_cpu = X.detach().cpu()
            del X
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Pick a conservative tile size based on free mem
            free_gb = _estimate_free_mem_gb()
            if free_gb >= 4.0:
                tile_size = 160
            elif free_gb >= 2.0:
                tile_size = 128
            else:
                tile_size = 96

            print(f"Available GPU memory: {free_gb:.2f} GB")
            print(f"Using tile size: {tile_size}x{tile_size}x{tile_size}\n")

            return apply_nonlocal_means_3d_tiled(
                X_cpu,
                kernel_size=kernel_size,
                std=std,
                kernel_size_mean=kernel_size_mean,
                tile_size=tile_size,
                debug=debug
            )
        else:
            raise


# Main export alias for 3D NLM with OOM-safe fallback
nlm3d = apply_nonlocal_means_3d_with_fallback