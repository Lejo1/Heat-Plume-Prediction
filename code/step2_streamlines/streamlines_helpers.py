import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import phi.torch.flow as pf

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
def build_velocity_grid(vx, vy, dims, randomK_data:bool=False):
    if randomK_data:
        u_axis0, u_axis1 = vy, vx
    else:
        u_axis0, u_axis1 = vx, vy
    # velocity samples at integer coordinates 0..N-1, same convention as the previous RegularGridInterpolator
    bounds = pf.Box(x=(-.5, dims[0]-.5), y=(-.5, dims[1]-.5))
    values = pf.math.wrap(np.stack([u_axis0, u_axis1], axis=-1), pf.spatial('x,y'), pf.channel(vector='x,y'))
    return pf.CenteredGrid(values, pf.extrapolation.BOUNDARY, bounds=bounds)

@pf.jit_compile(auxiliary_args='step_size')
def move_along_field(x, velocity, step_size):
    return pf.advect.points(pf.geom.Point(x), velocity, step_size, integrator=pf.advect.rk4).center

def calc_streamlines(start_points, velocity, maxs_xy, t_end=27.5, t_steps=1000, max_step_cells=0.5):
    # Solve for all start points at once. The RK4 step count is chosen so that no point moves
    # more than max_step_cells per step; the coarse solution is then linearly upsampled to
    # t_steps samples for drawing.
    starts = pf.math.wrap(np.asarray(start_points, dtype=np.float32), pf.instance('start_point'), pf.channel(vector='x,y'))
    v_max = float(pf.math.max(pf.math.norm(velocity.values, 'vector')))
    n_int = max(min(int(np.ceil(t_end * v_max / max_step_cells)), t_steps - 1), 16)
    step_size = t_end / n_int

    trajectory = pf.iterate(lambda x: move_along_field(x, velocity, step_size), pf.spatial(iter=n_int), starts)
    trajectory = trajectory.numpy('start_point,iter,vector')
    t_coarse = np.linspace(0, t_end, n_int + 1)
    t = np.linspace(0, t_end, t_steps)

    streamlines = []
    for line in trajectory:
        sol_x = np.interp(t, t_coarse, line[:,0])
        sol_y = np.interp(t, t_coarse, line[:,1])
        # cut each line where it first leaves the domain (0..x.max(), 0..y.max())
        inside = (sol_x >= 0) & (sol_x <= maxs_xy[0]) & (sol_y >= 0) & (sol_y <= maxs_xy[1])
        length = len(t) if inside.all() else np.argmin(inside)
        streamlines.append((sol_x[:length], sol_y[:length], t[:length]))
    return streamlines

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

def make_streamlines(mat_ids, vx, vy, dims, offset:str=None, randomK_data:bool=False, faded:bool=True, **kwargs):
    pos_hps = np.array(np.where(mat_ids == 2)).T.astype(float)
    print("Number of heat pumps: ", pos_hps.shape[0])
    if pos_hps.shape[0] == 0:
        return torch.tensor(np.zeros(dims))
    pos_hps += np.array([0.5,0.5]) # cell-center offset
    resolution = 5
    if offset != None:
        pos_hps += np.array([0,offset])

    time = datetime.now()
    velocity = build_velocity_grid(vx/resolution, vy/resolution, dims, randomK_data=randomK_data)
    streamlines = calc_streamlines(pos_hps, velocity, (dims[0]-1, dims[1]-1), t_end=27.5, t_steps=10_000)
    print("Time for calculating streamlines: ", datetime.now()-time, " seconds")

    streamlines_drawn = draw_streamlines(np.zeros(dims), streamlines, faded=faded)

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
