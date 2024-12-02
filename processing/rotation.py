import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from itertools import product, repeat
import math

# function to rotate one datapoint counter-clockwise (with pressure as input)
def rotate(data : torch.tensor, angle : int) -> torch.tensor:
    data_out = torch.zeros_like(data)
    # rotate all scalar fields
    for i in range(data.shape[0]):
        data_out[i] = TF.rotate(data[i].unsqueeze(0), angle, interpolation = InterpolationMode.BILINEAR).squeeze(0) #interpolation = InterpolationMode.BILINEAR
    
    return data_out

# rotate a datapoint such that direction matches specified direction and return rerotated prediction (with pressure as input)
def rotate_and_infer(datapoint : torch.tensor, grad_vec : list, model : torch.nn.Module, info, device : str) -> torch.tensor:
    # calculate gradient and get angle for aligning data point
    angle = get_rotation_angle(get_pressure_grad(datapoint,info), grad_vec)
    x = rotate(datapoint, angle)

    # get inference
    x = x.to(device).unsqueeze(0)
    y_out = model(x).to(device)

    # rotate result back
    y_out = rotate(y_out.cpu().detach(), 360 - angle)
    return y_out

# rotate a batch such that direction matches specified direction and return rerotated inference (with pressure as input)
def rotate_and_infer_batch(batch : torch.tensor, grad_vec : list, model : torch.nn.Module, info, device : str) -> torch.tensor:
    y_out_list = []
    
    for datapoint in batch:
        #calculate gradient and get angle for aligning data point
        angle = get_rotation_angle(get_pressure_grad(datapoint,info), grad_vec)
        x = rotate(datapoint, angle)

        #get inference
        x = x.to(device).unsqueeze(0)
        y_out = model(x).to(device)

        #rotate result back
        y_out = rotate(y_out.cpu().detach(), 360 - angle).squeeze(0)
        y_out_list.append(y_out)
    return torch.stack(y_out_list)

# get angle to rotate a counter-clockwise to match b's direction
def get_rotation_angle(a : list,b :list) -> int:
    # calculate the dot product and the determinant
    dot_product = np.dot(a, b)
    determinant = a[0] * b[1] - a[1] * b[0]
    
    # calculate the angle
    angle = np.degrees(np.arctan2(determinant, dot_product))
    
    # turn angle positive if necessary
    if angle < 0:
        angle += 360
    
    return angle

# get pressure gradient encoded by the data points pressure field
def get_pressure_grad(datapoint : torch.tensor, info) -> list:
    
    # get indices for calculating gradient and leave border of size 5 so that masks dont interfere
    p_ind = info['Inputs']['Liquid Pressure [Pa]']['index']
    center = int(datapoint[p_ind].shape[0]/2)
    start = 5
    end = datapoint[p_ind].shape[0] - 5
    dif = end - start

    # calculate gradient
    return [(datapoint[p_ind][end][center].item() - datapoint[p_ind][start][center].item())/dif, 
                                (datapoint[p_ind][center][end].item() - datapoint[p_ind][center][start].item())/dif]

# build mask to cut out circular field from input, based on:
# https://quva-lab.github.io/escnn/api/escnn.nn.html?highlight=maskmodule#escnn.nn.MaskModule
# returns tensor with just 0 and 1
def build_mask(s : int, dim: int = 2, dtype=torch.float32) -> torch.tensor:
    mask = torch.zeros(1, 1, *repeat(s, dim), dtype=dtype)
    c = (s-1) / 2  # center of the tensor
    r_max = c**2  # maximum radius squared for the circle to fit

    for k in product(range(s), repeat=dim):
        r = sum((x - c)**2 for x in k)
        if r <= r_max:
            mask[(..., *k)] = 1.  # inside the circle
        else:
            mask[(..., *k)] = 0.  # outside the circle
    return mask

# get number of cells inside circle described by mask
def mask_size(s, dim: int = 2) -> int:
    c = (s-1) / 2  # center of the tensor
    r_max = c**2  # maximum radius squared for the circle to fit
    pixels = 0.

    for k in product(range(s), repeat=dim):
        r = sum((x - c)**2 for x in k)
        if r <= r_max:
            pixels += 1.  # inside the circle
    return pixels

# cut out circular field of data
def mask_tensor(data : torch.tensor) -> torch.tensor:
    data_out = torch.zeros_like(data)
    mask = build_mask(data.shape[1], dtype = data.dtype)
    
    #apply mask to each channel
    for i in range(data.shape[0]):
        data_out[i] = data[i]*mask
    
    return data_out

#cut out circular field of data for batch
def mask_batch(batch : torch.tensor) -> torch.tensor:
    return torch.stack([mask_tensor(data) for data in batch])