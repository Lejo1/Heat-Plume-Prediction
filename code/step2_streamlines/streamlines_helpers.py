import numpy as np
from tqdm import tqdm
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp
from datetime import datetime
from pathlib import Path

from utils.utils_args import load_yaml, save_yaml


def build_new_args(args, model_name: str = None):
    args["inputs"] = [
    "Material ID",
    "Liquid X-Velocity [m_per_y]",
    "Liquid Y-Velocity [m_per_y]",
    "Streamlines Faded [-]",
    "Permeability X [m^2]",
    "Streamlines Faded Outer [-]"
    ]
    args["outputs"] = ["Temperature [C]"]
    if model_name:
        idx_vx = 1
        idx_vy = 2
        args["inputs"][idx_vx] = f"Liquid X-Velocity [m_per_y] - predicted by '{model_name.name}'"
        args["inputs"][idx_vy] = f"Liquid Y-Velocity [m_per_y] - predicted by '{model_name.name}'"

def build_new_info(info:dict, info_vx:dict, info_vy:dict):
    info["Inputs"]["Streamlines Faded [-]"] = {
    "index": 3,
    "max": 1.0,
    "mean": None,
    "min": 0.0,
    "norm": None,
    "std": None,
    }
    info["Inputs"]["Streamlines Faded Outer [-]"] = {
    "index": 5,
    "max": 1.0,
    "mean": None,
    "min": 0.0,
    "norm": None,
    "std": None,
    }
    info["Inputs"]["Permeability X [m^2]"]["index"] = 4

    info["Inputs"]["Liquid X-Velocity [m_per_y]"] = info_vx
    info["Inputs"]["Liquid X-Velocity [m_per_y]"]["index"] = 1
    info["Inputs"]["Liquid Y-Velocity [m_per_y]"] = info_vy
    info["Inputs"]["Liquid Y-Velocity [m_per_y]"]["index"] = 2

# data processing + streamline calculation
def integrate_velocity(x, y, vx, vy, randomK_data:bool=False):
        if randomK_data:
            fx = RegularGridInterpolator((x,y), vx, bounds_error=False, fill_value=None, method="linear")
            fy = RegularGridInterpolator((x,y), vy, bounds_error=False, fill_value=None, method="linear")
        else:
            fy = RegularGridInterpolator((x,y), vx, bounds_error=False, fill_value=None, method="linear")
            fx = RegularGridInterpolator((x,y), vy, bounds_error=False, fill_value=None, method="linear")

        # defines the velocity function to be integrated
        def f(t, y):
            return np.squeeze([fy(y), fx(y)])

        return f

def calc_streamline(interpolator, maxs_xy, start=[5,14], t_end=27.5, t_steps=1000, **kwargs):
    # Solve for start point
    sol = solve_ivp(interpolator, [0, t_end], start, t_eval=np.linspace(0,t_end,t_steps), **kwargs)

    sol_x = sol.y[0]
    sol_y = sol.y[1]

    # cut sol_y, sol_x if extend x.max(),y.max() or x.min(),y.min() (= 0,0)
    sol_x = sol_x[sol_x <= maxs_xy[0]]
    sol_y = sol_y[sol_y <= maxs_xy[1]]
    sol_x = sol_x[sol_x >= 0]
    sol_y = sol_y[sol_y >= 0]

    length = np.min([sol_x.shape[0], sol_y.shape[0]])
    sol_x = sol_x[:length]
    sol_y = sol_y[:length]
    t = sol.t[:length]
    
    return sol_x, sol_y, t

def draw_streamlines(image_data:np.array, streamlines:list, faded:bool=False):
    time = datetime.now()
    for streamline in streamlines:
        if faded and len(streamline[2]) > 0:
            val = streamline[2][::-1]
            val = val / val.max()
        else:
            val = 1
        image_data[((streamline[0]+0.5).astype(int),(streamline[1]+0.5).astype(int))] = val
    print("Time for drawing streamlines: ", datetime.now()-time, " seconds")
    return image_data

def make_streamlines(mat_ids, vx, vy, dims, offset:str=None, randomK_data:bool=False, **kwargs):
    pos_hps = np.array(np.where(mat_ids == 2)).T.astype(float)
    print("Number of heat pumps: ", pos_hps.shape[0])
    pos_hps += np.array([0.5,0.5]) # cell-center offset
    resolution = 5
    if offset != None:
        pos_hps += np.array([0,offset])
    x,y = (np.arange(0,dims[0]),np.arange(0,dims[1]))

    time = datetime.now()
    streamlines = []
    integrator = integrate_velocity(x, y, vx/resolution, vy/resolution, randomK_data=randomK_data)
    for hp in tqdm(pos_hps, desc="Calculating streamlines"):
        sol = calc_streamline(integrator, (x.max(),y.max()), np.array(hp).T, t_end=27.5, t_steps=10_000, **kwargs)
        streamlines.append(sol)
    print("Time for calculating streamlines: ", datetime.now()-time, " seconds")

    # streamlines_drawn = draw_streamlines(np.zeros(dims), streamlines, faded=False) # TODO for exp_inputs
    streamlines_drawn = draw_streamlines(np.zeros(dims), streamlines, faded=True)

    return torch.tensor(streamlines_drawn)

def save_new_datapoint(destination, runid:str, inputs_new:torch.Tensor, labels_new:torch.Tensor=None):
    (destination/"Inputs").mkdir(exist_ok=True, parents=True)
    (destination/"Labels").mkdir(exist_ok=True, parents=True)
    torch.save(inputs_new, destination / "Inputs" / runid)
    if labels_new != None:
        torch.save(labels_new, destination / "Labels" / runid)

def correct_args_info(destination:Path, v_info_path:Path=None, based_on_predicted_v:bool=False):
    info = load_yaml(destination / "info.yaml")
    info_v = load_yaml(v_info_path / "info.yaml")
    info_vx = info_v["Labels"]["Liquid X-Velocity [m_per_y]"]
    info_vy = info_v["Labels"]["Liquid Y-Velocity [m_per_y]"]
    build_new_info(info, info_vx, info_vy)
    save_yaml(info, destination / "info.yaml")

    args = load_yaml(destination / "args.yaml")
    if based_on_predicted_v:
        build_new_args(args, v_info_path)
    else:
        build_new_args(args)
    save_yaml(args, destination / "args.yaml")

def extend_inputs_dims(inputs_normed):
    dummy_field = torch.zeros_like(inputs_normed[0])
    inputs_new = torch.cat([inputs_normed[:3], dummy_field.unsqueeze(0), inputs_normed[3].unsqueeze(0), dummy_field.unsqueeze(0)], dim=0)
    inputs_new = inputs_new.float()

    return inputs_new
