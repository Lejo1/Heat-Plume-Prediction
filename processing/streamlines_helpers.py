import numpy as np
from copy import deepcopy
from tqdm import tqdm
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp
from datetime import datetime
import pathlib

from preprocessing.transforms import NormalizeTransform
from utils.utils_args import load_yaml, save_yaml, make_paths
import preprocessing.preprocessing as prep
from main import read_cla

# functions for data loading + preparing
## prepare data for inputs for streamlines: ik_Txy
def prepare_ik_Txy(dataset_name:str, problem:str="allin1"):
    args = read_cla(pathlib.Path("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/allin1/dummy_prep_ik_Txy"))
    args["data_raw"] = f"/scratch/sgs/pelzerja/datasets/{problem}/{dataset_name}"
    args["data_prep"] = f"/scratch/sgs/pelzerja/datasets_prepared/{problem}/{dataset_name} inputs_ik outputs_Txy"

    make_paths(args) # and check if data / model exists
    prep.preprocessing(args) # and save info.yaml in model folder

## prepare inputs for streamlines: ixyk_T
def load_data(dataset_name, runid, problem:str="allin1"):
    info = load_yaml(f"/scratch/sgs/pelzerja/datasets_prepared/{problem}/{dataset_name} inputs_ik outputs_Txy/info.yaml")
    norm = NormalizeTransform(info)

    inputs = torch.load(f"/scratch/sgs/pelzerja/datasets_prepared/{problem}/{dataset_name} inputs_ik outputs_Txy/Inputs/{runid}")
    inputs_normed = deepcopy(inputs)
    norm.reverse(inputs, "Inputs")

    labels = torch.load(f"/scratch/sgs/pelzerja/datasets_prepared/{problem}/{dataset_name} inputs_ik outputs_Txy/Labels/{runid}")
    labels_normed = deepcopy(labels)
    norm.reverse(labels, "Labels")

    args = load_yaml(f"/scratch/sgs/pelzerja/datasets_prepared/{problem}/{dataset_name} inputs_ik outputs_Txy/args.yaml")
    print("current:", labels.shape, inputs_normed.shape)

    return inputs, inputs_normed, labels, labels_normed, info, args

def extend_inputs_dims(inputs_normed):
    dummy_field = torch.zeros_like(inputs_normed[0])
    inputs_new = torch.cat([inputs_normed[:3], dummy_field.unsqueeze(0), inputs_normed[3].unsqueeze(0), dummy_field.unsqueeze(0)], dim=0)
    inputs_new = inputs_new.float()

    return inputs_new

def build_new_inputs_and_outputs(inputs_normed, labels_normed, indices):
    dummy_field = torch.zeros_like(inputs_normed[0])

    inputs_new = torch.cat([inputs_normed[indices["mat_id"]].unsqueeze(0), labels_normed[indices["vx"]].unsqueeze(0), labels_normed[indices["vy"]].unsqueeze(0), dummy_field.unsqueeze(0), inputs_normed[indices["k"]].unsqueeze(0)], dim=0) #, dummy_field.unsqueeze(0)
    
    labels_new = labels_normed[indices["T"]].unsqueeze(0)
    print("new:", inputs_new.shape, labels_new.shape)
    # change inputs_new dtype to float32
    inputs_new = inputs_new.float()
    return inputs_new, labels_new

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

    # info["Labels"] = {"Temperature [C]": info["Labels"]["Temperature [C]"]}

def extract_mat_ids_and_velos(inputs, label, indices):
    mat_ids = inputs[indices["mat_id"]].numpy()
    vx_real = label[indices["vx"]].numpy()
    vy_real = label[indices["vy"]].numpy()
    return mat_ids, vx_real, vy_real

def extract_required_data(dataset_name, runid, problem):
    indices = {
        "mat_id": 0,# input
        "k": 1,     # input
        "T": 0,     # label
        "vx": 1,    # label
        "vy": 2,    # label
    }

    inputs, inputs_normed, label, label_normed, info, args = load_data(dataset_name, runid, problem)
    mat_ids, vx_real, vy_real = extract_mat_ids_and_velos(inputs, label, indices)
    
    dims = mat_ids.shape

    return inputs_normed, label_normed, args, info, mat_ids, vx_real, vy_real, dims, indices

# data processing + streamline calculation
def integrate_velocity(x, y, vx, vy, dummy_data:bool=False):
        if dummy_data:
            print("Dummy data")
            # exit()
            fx = RegularGridInterpolator((x,y), vx, bounds_error=False, fill_value=None, method="linear")
            fy = RegularGridInterpolator((x,y), vy, bounds_error=False, fill_value=None, method="linear")
        else:
            fy = RegularGridInterpolator((x,y), vx, bounds_error=False, fill_value=None, method="linear")
            fx = RegularGridInterpolator((x,y), vy, bounds_error=False, fill_value=None, method="linear")

        # define the velocity function to be integrated:
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

def draw_streamlines(image_data:np.array, streamlines:list, faded:bool=False, t_end:float=27.5):
    time = datetime.now()
    for streamline in streamlines:
        if faded and len(streamline[2]) > 0:
            # TODO why happen that streamline of length=0?
            val = (t_end - streamline[2]) / t_end
        else:
            val = 1
        image_data[((streamline[0]+0.5).astype(int),(streamline[1]+0.5).astype(int))] = np.maximum(image_data[((streamline[0]+0.5).astype(int),(streamline[1]+0.5).astype(int))], val) # cell-center offset
    print("Time for drawing streamlines: ", datetime.now()-time, " seconds")
    return image_data

def draw_streamlines_gauss(field_max, field_mean, streamlines:list, n_streams_per_hp, faded:bool=False, t_end:float=27.5):
    time = datetime.now()
    for i in range(0, len(streamlines), n_streams_per_hp):
        dataslice = streamlines[i:i+n_streams_per_hp]
        local_mean = np.zeros(field_mean.shape)

        for streamline in dataslice:
            if faded and len(streamline[2]) > 0:
                val = (t_end - streamline[2]) / t_end
            else:
                val = 1
            
            field_max[((streamline[0]+0.5).astype(int),(streamline[1]+0.5).astype(int))] = np.maximum(field_max[((streamline[0]+0.5).astype(int),(streamline[1]+0.5).astype(int))], val) # +0.5 := cell-center offset
            local_mean[((streamline[0]+0.5).astype(int),(streamline[1]+0.5).astype(int))] += 1/len(dataslice)
        
        field_mean = np.maximum(field_mean, local_mean)

    # rescale t0 (0,1)
    field_mean_min = np.min(field_mean)
    field_mean_max = np.max(field_mean)
    field_mean = (field_mean - field_mean_min) / (field_mean_max - field_mean_min)

    print("Time for drawing streamlines: ", datetime.now()-time, " seconds")
    return field_max, field_mean

def make_streamlines(mat_ids, vx, vy, dims, offset:str=None, dummy_data:bool=False, **kwargs):
    resolution = 5
    x,y = (np.arange(0,dims[0]),np.arange(0,dims[1]))

    pos_hps = np.array(np.where(mat_ids == 2)).T.astype(float)
    # pos_hps += np.array([0.5,0.5]) # cell-center offset
    
    n_streamlines_per_hp = 50
    if offset == "gauss":
        gauss_pos_hps = []
        for hp in pos_hps:
            hp_i_gauss = gaussian_distribution(center=hp, stddev=[5,5], num_points=n_streamlines_per_hp, x_min=[hp[0],0], x_max=[hp[0],dims[0]])
            gauss_pos_hps.append(hp_i_gauss.T)
        gauss_pos_hps = np.array(gauss_pos_hps)
        pos_hps = gauss_pos_hps.reshape(-1,2)
    elif offset != None:
        pos_hps += np.array([0,offset])
    
    time = datetime.now()
    streamlines = []
    integrator = integrate_velocity(x, y, vx/resolution, vy/resolution, dummy_data=dummy_data)
    t_end = 27.5
    for hp in tqdm(pos_hps, desc="Calculating streamlines"):
        sol = calc_streamline(integrator, (x.max(),y.max()), np.array(hp).T, t_end=t_end, t_steps=10_000, **kwargs)
        streamlines.append(sol)
    print("Time for calculating streamlines: ", datetime.now()-time, " seconds")
    if offset != "gauss":
        streamlines_drawn = draw_streamlines(np.zeros(dims), streamlines, faded=True, t_end=t_end)
        return torch.tensor(streamlines_drawn)
    else:
        streamlines_drawn_0, streamlines_drawn_1 = draw_streamlines_gauss(np.zeros(dims), np.zeros(dims), streamlines, n_streams_per_hp=n_streamlines_per_hp, faded=True, t_end=t_end) 
        return torch.tensor(streamlines_drawn_0), torch.tensor(streamlines_drawn_1)


# save new data
def save_new_datapoint(destination, runid:str, inputs_new:torch.Tensor, labels_new:torch.Tensor=None):
    (destination/"Inputs").mkdir(exist_ok=True, parents=True)
    (destination/"Labels").mkdir(exist_ok=True, parents=True)
    torch.save(inputs_new, destination / "Inputs" / runid)
    if labels_new != None:
        torch.save(labels_new, destination / "Labels" / runid)

def correct_args_info(destination:pathlib.Path, v_info_path:pathlib.Path=None, based_on_predicted_v:bool=False):
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

def gaussian_distribution(center:list, stddev:list, num_points:int, x_min:list, x_max:list):
    # return x values distributed according to a Gaussian centered at 'center' with standard deviation 'stddev'
    samples = []
    for i in range(len(center)):
        x_values = np.linspace(x_min[i], x_max[i], 1000)
        gaussian = np.exp(-0.5 * ((x_values - center[i]) / stddev[i]) ** 2)
        gaussian /= np.sum(gaussian)  # Normalize to create a probability distribution
        samples.append(np.random.choice(x_values, size=num_points, p=gaussian/np.sum(gaussian)))
    return np.array(samples)