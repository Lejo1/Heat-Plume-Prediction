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

def build_new_info(info:dict, i_s:int=3, i_s_outer:int=5, i_s_seasonal:int=None):
    info["Inputs"]["Streamlines Faded [-]"] = {
    "index": i_s,
    "max": 1.0,
    "mean": None,
    "min": 0.0,
    "norm": None,
    "std": None,
    }
    info["Inputs"]["Streamlines Faded Outer [-]"] = {
    "index": i_s_outer,
    "max": 1.0,
    "mean": None,
    "min": 0.0,
    "norm": None,
    "std": None,
    }
    if i_s_seasonal is not None:
        info["Inputs"]["Streamlines Faded Seasonal [-]"] = {
        "index": i_s_seasonal,
        "max": 1.0,
        "mean": None,
        "min": 0.0,
        "norm": None,
        "std": None,
        }
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

def draw_streamlines(image_data:np.array, streamlines:list, faded:bool=False, t_end:float=27.5, seasonal:bool=False):
    time = datetime.now()
    for streamline in streamlines:
        if faded and len(streamline[2]) > 0:
            # TODO why happen that streamline of length=0?
            value = (t_end - streamline[2]) / t_end
        else:
            value = 1

        if seasonal:
            streamline_t = streamline[2] % 365
            value = np.where(streamline_t < 365/4, value, np.where(streamline_t < 365/2, -value, np.where(streamline_t < 365/4*3, value, -value))) # fade in winter, fade out in summer
            value = (value + 1).astype(float) / 2 + 0.5

        image_data[((streamline[0]+0.5).astype(int),(streamline[1]+0.5).astype(int))] = np.maximum(image_data[((streamline[0]+0.5).astype(int),(streamline[1]+0.5).astype(int))], value) # cell-center offset
    print("Time for drawing streamlines: ", datetime.now()-time, " seconds")
    return image_data

def make_streamlines(mat_ids, vx, vy, dims, runid, offset:str=None, randomK_data:bool=False, faded: bool=True, seasonal:bool=False, **kwargs):
    # pos_hps = np.array(np.where(mat_ids == 2)).T.astype(float)
    pos_hps = np.array(np.where(mat_ids < 0)).T.astype(float)
    print("Number of heat pumps: ", pos_hps.shape[0])
    resolution = 2 #5
    if offset != None:
        if randomK_data:
            pos_hps += np.array([0,offset])
        else:
            pos_hps += np.array([offset,0])
    x,y = (np.arange(0,dims[0]),np.arange(0,dims[1]))

    time = datetime.now()
    streamlines = []
    
    integrator = integrate_velocity(x, y, vx/resolution, vy/resolution, randomK_data=randomK_data)
    t_end = 10*365 #27.5 # adaptation because velocities not in [m/y] but in [m/d] now
    for hp in tqdm(pos_hps, desc="Calculating streamlines"):
        # sol = calc_streamline(integrator, (x.max(),y.max()), np.array(hp).T, t_end=27.5, t_steps=10_000, **kwargs)
        sol = calc_streamline(integrator, (x.max(),y.max()), np.array(hp).T, t_end=t_end, t_steps=t_end//5, **kwargs) #tsteps: 10y*365d a one step every 5days
        streamlines.append(sol)
    print("Time for calculating streamlines: ", datetime.now()-time, " seconds")

    # streamlines_drawn = draw_streamlines(np.zeros(dims), streamlines, faded=False) # TODO for exp_inputs
    streamlines_drawn = draw_streamlines(np.zeros(dims), streamlines, faded=faded, t_end=t_end, seasonal=seasonal)

    return torch.tensor(streamlines_drawn)

def save_new_datapoint(destination, runid:str, inputs_new:torch.Tensor, labels_new:torch.Tensor=None):
    (destination/"Inputs").mkdir(exist_ok=True, parents=True)
    (destination/"Labels").mkdir(exist_ok=True, parents=True)
    torch.save(inputs_new, destination / "Inputs" / runid)
    if labels_new != None:
        torch.save(labels_new, destination / "Labels" / runid)

def correct_args_info(destination:Path, v_info_path:Path=None, based_on_predicted_v:bool=False, i_s:int=3, i_s_outer:int=5, i_s_seasonal:int=None):
    info = load_yaml(destination / "info.yaml")
    build_new_info(info, i_s, i_s_outer, i_s_seasonal)
    save_yaml(info, destination / "info.yaml")

    # args = load_yaml(destination / "args.yaml")
    # if based_on_predicted_v:
    #     build_new_args(args, v_info_path)
    # else:
    #     build_new_args(args)
    # save_yaml(args, destination / "args.yaml")

def extend_inputs_dims_specific(inputs_normed):
    dummy_field = torch.zeros_like(inputs_normed[0])
    inputs_new = torch.cat([inputs_normed[:3], dummy_field.unsqueeze(0), inputs_normed[3].unsqueeze(0), dummy_field.unsqueeze(0)], dim=0)
    inputs_new = inputs_new.float()

    return inputs_new

def extend_inputs_dims(inputs_normed, n_added_inputs:int=2):
    inputs_new = torch.zeros(len(inputs_normed)+n_added_inputs, *inputs_normed[0].shape)
    inputs_new[:len(inputs_normed)] = inputs_normed
    inputs_new = inputs_new.float()
    return inputs_new