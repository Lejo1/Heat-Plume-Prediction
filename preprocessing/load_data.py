import h5py
import numpy as np
import torch
from typing import Tuple
from pathlib import Path
import matplotlib.pyplot as plt

def load_data(data_path: Path, time: str, variables: dict, dimensions_of_datapoint: tuple, additional_input: torch.Tensor = None, time_prediction: str= "   1 Time  2.75000E+01 y", print_bool: bool = False, refined:bool=False, goal_resolution:float=None):
    """
    Load data from h5 file on data_path, but only the variables named in variables.get_ids() at time stamp variables.time
    Sets the values of each PhysicalVariable in variables to the loaded data.
    """
    if refined:
        fct_reshape = lambda x,key: reshape_refined(data_path.parent, x, dimensions_of_datapoint, goal_resolution, key)
    else:
        fct_reshape = lambda x, key: x.reshape(dimensions_of_datapoint)
    data = dict()
    with h5py.File(data_path, "r") as file:
        for key in variables:  # properties
            try:
                if "velocity" in key.lower():
                    data[key] = torch.tensor(fct_reshape(np.array(file[time_prediction][key]), key)).float()
                else:
                    data[key] = torch.tensor(fct_reshape(np.array(file[time][key]), key)).float()
            except KeyError:
                if key == "SDF":
                    data[key] = torch.tensor(fct_reshape(np.array(file[time]["Material ID"]), "Material ID")).float()
                elif key in ["PE x", "PE y", "MDF", "LST"]:
                    data[key] = torch.tensor(fct_reshape(np.array(file[time][key]), key)).float()
                elif key == "Pressure Gradient [-]":
                    empty_field = torch.ones(list(dimensions_of_datapoint)).float()
                    pressure_grad = get_pressure_gradient(data_path)
                    data[key] = empty_field * pressure_grad
                elif key == "Original Temperature [C]":
                    empty_field = torch.zeros(list(dimensions_of_datapoint)).float()
                    data[key] = empty_field #*10.6
                elif key == "Preprocessed Temperature [C]":
                    data[key] = additional_input.float()
                    assert additional_input.shape == tuple(dimensions_of_datapoint), f"{additional_input.shape=}, {dimensions_of_datapoint=}"
                else:
                    if "velocity" in key: timi = time_prediction
                    else: timi = time
                    raise KeyError(f"Key '{key}' not found in {data_path} at time {timi}")
            if print_bool:
                if "velocity" in key: timi = time_prediction
                else: timi = time
                print(f"Loaded {key} at time {timi} with shape {data[key].shape}")
    return data

def reshape_refined(run_dir: Path, data: torch.Tensor, box_dims: Tuple, goal_resolution: float, key:str):
    """
    Reshape the data to the goal resolution
    """
    if "Material ID" in key:
        method = "max"
    else:
        method = "mean"

    mesh_and_res = load_mesh(run_dir)
    assert data.shape[0] == mesh_and_res.shape[0], f"{data.shape=}, {mesh_and_res.shape=}"
    values = generate_regular_cell_values(goal_resolution, mesh_and_res, data, box_dims, method)
    return values

def load_mesh(path_run: Path):
    with h5py.File(path_run/"mesh.h5", "r") as mesh_file:
        cell_centers = np.array(mesh_file["Domain/Cells/Centers"])
        cell_volumes = np.array(mesh_file["Domain/Cells/Volumes"])
        mesh = np.concatenate([cell_centers, cell_volumes[:,None]], axis=1)

        mesh[:, 3] = np.round(np.cbrt(mesh[:, 3]),8)
    return mesh

def generate_regular_cell_values(plot_res: float, mesh: np.ndarray, data:np.ndarray, n_cells: Tuple[int], method:str="mean") -> np.ndarray:
    # interpolate and average data to mesh
    # z-dimension: always averaged
    # method: "mean" or "max"

    values = np.zeros((n_cells[0], n_cells[1]))

    if method == "mean":
        weights = np.zeros((n_cells[0], n_cells[1]))
        for (curr_x,curr_y,curr_z,curr_res), value in zip(mesh, data):
            start_pos = np.array([curr_x, curr_y])-curr_res/2
            cell = (start_pos/plot_res).astype(int)
            if curr_res <= plot_res:
                values[cell[0], cell[1]] += value * (curr_res/plot_res)**3
                weights[cell[0], cell[1]] += (curr_res/plot_res)**3
            elif curr_res > plot_res:
                # update all cells that are covered by the larger cell
                for i in range(int(curr_res/plot_res)):
                    for j in range(int(curr_res/plot_res)):
                        values[cell[0]+i, cell[1]+j] += value
                        weights[cell[0]+i, cell[1]+j] += 1
            else:
                raise ValueError(f"{curr_res=}, {plot_res=}")
        values /= weights
        
    elif method == "max":
        for (curr_x,curr_y,curr_z,curr_res), value in zip(mesh, data):
            start_pos = np.array([curr_x, curr_y])-curr_res/2
            cell = (start_pos/plot_res).astype(int)
            if curr_res <= plot_res:
                values[cell[0], cell[1]] = max(values[cell[0], cell[1]], value)
            elif curr_res > plot_res:
                # update all cells that are covered by the larger cell
                for i in range(int(curr_res/plot_res)):
                    for j in range(int(curr_res/plot_res)):
                        values[cell[0]+i, cell[1]+j] = max(values[cell[0]+i, cell[1]+j], value)
            else:
                raise ValueError(f"{curr_res=}, {plot_res=}")
    if len(values.shape) == 2:
        values = np.expand_dims(values, -1)
    return values

def get_hp_location(data): # TODO doublingg with transforms
    try:
        ids = data["Material ID"]
    except:
        try:
            ids = data["SDF"]
        except:
            try:
                ids = data["MDF"]
            except:
                return None
    max_id = ids.max()
    loc_hp = np.array(np.where(ids == max_id)).squeeze()
    return loc_hp

def get_hp_location_from_tensor(data: torch.Tensor, info: dict): # TODO doublingg with transforms
    try:
        idx = info["Inputs"]["Material ID"]["index"]
    except:
        idx = info["Inputs"]["SDF"]["index"]
    loc_hp = torch.Tensor(torch.where(data[idx] == data[idx].max())).squeeze().int().tolist()
    return loc_hp

def get_pressure_gradient(data_path):
        # ATTENTION! pressure switched axis
    try: #new data (2025)
        pressure_grad_file = data_path.parent / "bc_initial.txt"
        with open(pressure_grad_file, "r") as f:
            pressure_grad = f.read().split()[1:]
        pressure_grad = torch.tensor([float(grad) for grad in pressure_grad])
        pressure_grad = pressure_grad[0] # only x-direction

    except: #old data
        pressure_grad_file = data_path.parent / "pressure_gradient.txt"
        pressure_grad_file_interim = data_path.parent / "interim_pressure_gradient.txt"
        try:
            with open(pressure_grad_file, "r") as f:
                pressure_grad = f.read().split()[1:]
        except FileNotFoundError:
            with open(pressure_grad_file_interim, "r") as f:
                pressure_grad = f.read().split()[1:]
        pressure_grad = torch.tensor([float(grad) for grad in pressure_grad])
        pressure_grad = pressure_grad[1] # only y-direction
    assert pressure_grad != 0, f"{pressure_grad=} and it should not be zero"
    return pressure_grad


def detect_datapoints(dataset_path_raw: Path):
    """
    Create the simulation dataset by preparing a list of samples
    Simulation data are sorted in an ascending order by run number
    :returns: (data_paths, runs) where:
        - data_paths is a list containing paths to all simulation runs in the dataset, NOT the actual simulated data
        - runs is a list containing one label per run
    """
    set_data_paths_runs, runs = [], []
    found_dataset = False

    for folder in dataset_path_raw.iterdir():
        if folder.is_dir():
            for file in folder.iterdir():
                if file.name == "pflotran.h5":
                    set_data_paths_runs.append((folder.name, file))
                    found_dataset = True
    # Sort the data and runs in ascending order
    set_data_paths_runs = sorted(
        set_data_paths_runs, key=lambda val: int(val[0].strip('RUN_')))
    runs = [data_path[0] for data_path in set_data_paths_runs]
    data_paths = [data_path[1] for data_path in set_data_paths_runs]
    if not found_dataset:
        raise ValueError(f"No dataset found in {dataset_path_raw}")

    return data_paths, runs