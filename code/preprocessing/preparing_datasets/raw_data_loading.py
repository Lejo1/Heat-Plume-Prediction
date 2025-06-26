import h5py
import numpy as np
import torch
from pathlib import Path

def load_raw_data(data_path: Path, time: str, variables: dict, dimensions_of_datapoint: tuple, time_prediction: str, print_bool: bool = False):
    """
    Load data from h5 file on data_path, but only the variables named in variables.get_ids() at time stamp variables.time
    Sets the values of each PhysicalVariable in variables to the loaded data.
    """
    fct_reshape = lambda x: x.reshape(dimensions_of_datapoint)
    data = {}

    with h5py.File(data_path, "r") as file:
        for key in variables:  # properties
            try:
                if "velocity" in key.lower():
                    data[key] = torch.tensor(fct_reshape(np.array(file[time_prediction][key]))).float()
                else:
                    data[key] = torch.tensor(fct_reshape(np.array(file[time][key]))).float()
            except KeyError:
                if key == "Pressure Gradient [-]":
                    empty_field = torch.ones(list(dimensions_of_datapoint)).float()
                    pressure_grad = get_pressure_gradient(data_path)
                    data[key] = empty_field * pressure_grad
                else:
                    if "velocity" in key: timi = time_prediction
                    else: timi = time
                    raise KeyError(f"Key '{key}' not found in {data_path} at time {timi}")
            if print_bool:
                if "velocity" in key: timi = time_prediction
                else: timi = time
                print(f"Loaded {key} at time {timi} with shape {data[key].shape}")
    return data

def get_hp_location(data):
    try:
        ids = data["Material ID"]
    except:
        return None
    max_id = ids.max()
    loc_hp = np.array(np.where(ids == max_id)).squeeze()
    return loc_hp

def get_hp_location_from_tensor(data: torch.Tensor, info: dict): 
    idx = info["Inputs"]["Material ID"]["index"]
    loc_hp = torch.Tensor(torch.where(data[idx] == data[idx].max())).squeeze().int().tolist()
    return loc_hp

def get_pressure_gradient(data_path):
    try: #new data (2025)
        pressure_grad_file = data_path.parent / "bc_initial.txt"
        with open(pressure_grad_file, "r") as f:
            pressure_grad = f.read().split()[1:]
        pressure_grad = torch.tensor([float(grad) for grad in pressure_grad])
        pressure_grad = pressure_grad[0] # only x-direction

    except: #old data (pre2025)
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