import numpy as np
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt

from preprocessing.datasets.dataset import DataPoint
from processing.networks.unet import UNet
from postprocessing.visualization import reverse_norm_one_dp

def unify_size(inputs, required_size):
    start = [int((len(inputs[0]) - required_size[0])//2), int((len(inputs[0,0]) - required_size[1])//2)]
    inputs = inputs[:,start[0]:start[0]+required_size[0], start[1]:start[1]+required_size[1]]
    return inputs

def masking(data, threshold:float = 10.7):
    mask = data > threshold
    masked_data = data[mask]
    return mask, masked_data

# flood fill
def in_bounds(curr_pos, size):
    tmp = (curr_pos >= 0) & (curr_pos[0] < size[0]) & (curr_pos[1] < size[1])
    return tmp[0] & tmp[1]
 
def select_in_bounds(curr_pos, size):
    return curr_pos.T[in_bounds(curr_pos, size)].T
 
def filter_visited(curr_pos, visited, field):
    return curr_pos.T[(~visited[curr_pos[0],curr_pos[1]] & field[curr_pos[0],curr_pos[1]])].T
 
def step(active_indices, visited, field):
    size = visited.shape
    visited[active_indices[0],active_indices[1]] = True
    up = active_indices.copy()
    up[0] += 1
    down = active_indices.copy()
    down[0] -= 1
    right = active_indices.copy()
    right[1] += 1
    left = active_indices.copy()
    left[1] -= 1
    up = select_in_bounds(up, size)
    down = select_in_bounds(down, size)
    left = select_in_bounds(left, size)
    right = select_in_bounds(right, size)
    up = filter_visited(up, visited, field)
    visited[up[0],up[1]] = True
    down = filter_visited(down, visited, field)
    visited[down[0],down[1]] = True
    left = filter_visited(left, visited, field)
    visited[left[0],left[1]] = True
    right = filter_visited(right, visited, field)
    visited[right[0],right[1]] = True
 
    new_indices = np.concatenate((up, down, left, right), axis=1)
    return new_indices
 
def flood_fill(active_indices, field):
    visited=np.zeros_like(field,dtype=bool)
    i = 0
    while len(active_indices[0]) > 0:
        i += 1
        active_indices = step(active_indices, visited, field)
    return visited

def test_flood_fill():
    np.random.seed(0)
    size = 2560
    n_start = 100
    field_to_fill = np.ones((size, size), dtype=bool)
    active_indices = np.random.randint(0, size - 1, (2, n_start))
    flood_fill(active_indices, field_to_fill)

def connectivity_field_flood(mat_ids_unnormed, mask_output):
    hps = np.where(mat_ids_unnormed == 2)
    hps = np.array(hps)
    connectivity_field = flood_fill(hps, mask_output[0].cpu().numpy())

    unconnected_cells = connectivity_field ^ np.array(mask_output[0])
    connected_cells = np.sum(np.array(mask_output[0]))
    return connectivity_field, unconnected_cells, connected_cells

def calc_connectivity(model_path:Path, data_path:Path, data_id: int, model:UNet, id_mat_ids:int, threshold:float):
    # Data and model loading
    model.load(model_path)
    data = DataPoint(data_path, i=data_id)
    inputs, label = data[0]
    output = model.infer(inputs.unsqueeze(0))

    # Data preparation
    inputs, data, output = reverse_norm_one_dp(inputs, label, output, data.norm)
    required_size = [len(output[0]), len(output[0,0])]
    inputs = unify_size(inputs, required_size)
    label = unify_size(label, required_size)
    output = unify_size(output, required_size)

    conn_label_dict = connectivityLoss(inputs, label, id_mat_ids, threshold)
    conn_output_dict = connectivityLoss(inputs, output, id_mat_ids, threshold)
    conn_label_dict["field"] = label[0]
    conn_output_dict["field"] = output[0]

    return conn_label_dict, conn_output_dict

def connectivityLoss(inputs, data, id_mat_ids:int, threshold:float=10.7) -> Dict:
    """
    expects one data point 

    Returns: dict: array of connected cells, array of unconnected cells, ratio of unconnected/connected cells, i.e. no unit
    """
    # Data masking at threshold
    mask, _ = masking(data, threshold)

    connectivity, unconn_cells, conn_cells = connectivity_field_flood(inputs[int(id_mat_ids)], mask)
    ratio = np.sum(unconn_cells)/conn_cells
    unconn_percentage = np.sum(unconn_cells)/np.multiply(*inputs[int(id_mat_ids)].shape)*100
    
    return {"connectivity" : connectivity,
            "unconnected_cells" : unconn_cells,
            "unconnected_percentage" : unconn_percentage,
            "ratio" : ratio}