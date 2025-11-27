import time
from dataclasses import dataclass, field
from math import inf
from typing import Dict

# import matplotlib as mpl
# mpl.use('pgf')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader
from data_stuff.utils_3d import get_hp_position, get_hp_position_from_input, remove_first_dim

from data_stuff.transforms import NormalizeTransform
from networks.unet import UNet
from networks.unet3d import UNet3d

# mpl.rcParams.update({'figure.max_open_warning': 0})
# plt.rcParams['figure.figsize'] = [16, 5]

# TODO: look at vispy library for plotting 3D data

@dataclass
class DataToVisualize:
    data: np.ndarray
    name: str
    extent_highs :tuple = (1280,100) # x,y in meters
    imshowargs: Dict = field(default_factory=dict)
    contourfargs: Dict = field(default_factory=dict)
    contourargs: Dict = field(default_factory=dict)

    def __post_init__(self):
        extent = (0,int(self.extent_highs[0]),int(self.extent_highs[1]),0)

        self.imshowargs = {"cmap": "RdBu_r", 
                           "extent": extent}

        self.contourfargs = {"levels": np.arange(10.4, 16, 0.25), 
                             "cmap": "RdBu_r", 
                             "extent": extent}
        
        T_gwf = 10.6
        T_inj_diff = 5.0
        self.contourargs = {"levels" : [np.round(T_gwf + 1, 1)],
                            "cmap" : "Pastel1", 
                            "extent": extent}

        if self.name == "Liquid Pressure [Pa]":
            self.name = "Pressure in [Pa]"
        elif self.name == "Material ID":
            self.name = "Position of the heatpump in [-]"
        elif self.name == "Permeability X [m^2]":
            self.name = "Permeability in [m$^2$]"
        elif self.name == "SDF":
            self.name = "SDF-transformed position in [-]"
    
def visualizations(model: UNet | UNet3d, dataloader: DataLoader, device: str, amount_datapoints_to_visu: int = inf, plot_path: str = "default", pic_format: str = "png"):
    print("Visualizing...", end="\r")

    if amount_datapoints_to_visu > len(dataloader.dataset):
        amount_datapoints_to_visu = len(dataloader.dataset)

    norm = dataloader.dataset.dataset.norm
    info = dataloader.dataset.dataset.info
    model.to(device)
    model.eval()
    settings_pic = {"format": pic_format,
                    "dpi": 600,}

    current_id = 0
    for inputs, labels in dataloader:
        len_batch = inputs.shape[0]
        for datapoint_id in range(len_batch):
            name_pic = f"{plot_path}_{current_id}"

            x = torch.unsqueeze(inputs[datapoint_id].to(device), 0)
            y = labels[datapoint_id]
            y_out = model(x)

            x, y, y_out = reverse_norm_one_dp(x, y, y_out, norm)

            if y.dim()==3:# (X,Y,Z)
                # visualization for zy
                name_pic = f"{plot_path}_{current_id}_zy"
                y_label="z [m]"
                dict_to_plot = prepare_data_to_plot(x, y, y_out, info)
                plot_datafields(dict_to_plot, name_pic, settings_pic, y_label)

                # visualization for xy
                x=x.transpose(1,3)# (Z,Y,X)
                y=y.transpose(0,2)
                y_out=y_out.transpose(0,2)
                name_pic = f"{plot_path}_{current_id}_xy"
                
            y_label="x [m]"
            dict_to_plot = prepare_data_to_plot(x, y, y_out, info)
            plot_datafields(dict_to_plot, name_pic, settings_pic, y_label)
            # plot_isolines(dict_to_plot, name_pic, settings_pic)
            # measure_len_width_1K_isoline(dict_to_plot)

            if current_id >= amount_datapoints_to_visu-1:
                return None
            current_id += 1

def reverse_norm_one_dp(x: torch.Tensor, y: torch.Tensor, y_out:torch.Tensor, norm: NormalizeTransform):
    # reverse transform for plotting real values
    x = norm.reverse(x.detach().cpu().squeeze(0), "Inputs")
    y = norm.reverse(y.detach().cpu(),"Labels")[0]#[0] wie squeeze(0): 1,64,256,64 -> 64,256,64
    y_out = norm.reverse(y_out.detach().cpu()[0],"Labels")[0]
    return x, y, y_out

def prepare_data_to_plot(x: torch.Tensor, y: torch.Tensor, y_out:torch.Tensor, info: dict):
    # prepare data of temperature true, temperature out, error, physical variables (inputs)
    temp_max = max(y.max(), y_out.max())
    temp_min = min(y.min(), y_out.min())

    if len(y.shape) ==3:
        position=get_hp_position_from_input(x, info)[0]
        y=remove_first_dim(y, position)
        y_out=remove_first_dim(y_out, position)
        x=remove_first_dim(x, position)

    extent_highs = (np.array(info["CellsSize"][:2]) * x.shape[-2:])
    

    dict_to_plot = {
        "t_true": DataToVisualize(y, "Label: Temperature in [°C]", extent_highs, {"vmax": temp_max, "vmin": temp_min}),
        "t_out": DataToVisualize(y_out, "Prediction: Temperature in [°C]", extent_highs, {"vmax": temp_max, "vmin": temp_min}),
        "error": DataToVisualize(torch.abs(y-y_out), "Absolute error in [°C]", extent_highs),
    }
    inputs = info["Inputs"].keys()
    for input in inputs:
        index = info["Inputs"][input]["index"]
        dict_to_plot[input] = DataToVisualize(x[index], input, extent_highs)

    return dict_to_plot

def plot_datafields(data: Dict[str, DataToVisualize], name_pic: str, settings_pic: dict, y_label="x [m]"):
    # plot datafields (temperature true, temperature out, error, physical variables (inputs))

    num_subplots = len(data)
    fig, axes = plt.subplots(num_subplots, 1, sharex=True)
    fig.set_figheight(num_subplots)
    
    for index, (name, datapoint) in enumerate(data.items()):
        plt.sca(axes[index])
        plt.title(datapoint.name)
        # if name in ["t_true", "t_out"]:  
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("ignore")

        #         CS = plt.contour(torch.flip(datapoint.data, dims=[1]).T, **datapoint.contourargs)
        #     plt.clabel(CS, inline=1, fontsize=10)
        plt.imshow(datapoint.data.T, **datapoint.imshowargs)
        plt.gca().invert_yaxis()

        plt.ylabel(y_label)
        _aligned_colorbar()

    plt.sca(axes[-1])
    plt.xlabel("y [m]")
    plt.tight_layout()
    plt.savefig(f"{name_pic}.{settings_pic['format']}", **settings_pic)

def plot_isolines(data: Dict[str, DataToVisualize], name_pic: str, settings_pic: dict):
    # plot isolines of temperature fields
    num_subplots = 3 if "Original Temperature [C]" in data.keys() else 2
    fig, axes = plt.subplots(num_subplots, 1, sharex=True)
    fig.set_figheight(num_subplots)

    for index, name in enumerate(["t_true", "t_out", "Original Temperature [C]"]):
        try:
            plt.sca(axes[index])
            data[name].data = torch.flip(data[name].data, dims=[1])
            plt.title("Isolines of "+data[name].name)
            plt.contourf(data[name].data.T, **data[name].contourfargs)
            plt.ylabel("x [m]")
            _aligned_colorbar(ticks=[11.6, 15.6])
        except:
            pass

    plt.sca(axes[-1])
    plt.xlabel("y [m]")
    plt.tight_layout()
    plt.savefig(f"{name_pic}_isolines.{settings_pic['format']}", **settings_pic)

def infer_all_and_summed_pic(model: UNet | UNet3d, dataloader: DataLoader, device: str):
    '''
    sum inference time (including reverse-norming) and pixelwise error over all datapoints
    '''
    
    norm = dataloader.dataset.dataset.norm
    model.to(device)
    model.eval()

    current_id = 0
    avg_inference_time = 0
    summed_error_pic = torch.zeros_like(torch.Tensor(dataloader.dataset[0][0][0])).cpu()

    for inputs, labels in dataloader:
        len_batch = inputs.shape[0]
        for datapoint_id in range(len_batch):
            # get data
            x = inputs[datapoint_id].to(device)
            x = torch.unsqueeze(x, 0)

            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            y_out = model(x)
            end.record()
            torch.cuda.synchronize()
            avg_inference_time += start.elapsed_time(end)/1000
            
            y = labels[datapoint_id]

            # reverse transform for plotting real values
            x = norm.reverse(x.cpu().detach().squeeze(), "Inputs")
            y = norm.reverse(y.cpu().detach(),"Labels")[0]
            y_out = norm.reverse(y_out.cpu().detach()[0],"Labels")[0]
            summed_error_pic += abs(y-y_out)

            current_id += 1

    avg_inference_time /= current_id
    summed_error_pic /= current_id
    return avg_inference_time, summed_error_pic

def plot_avg_error_cellwise(dataloader, summed_error_pic, settings_pic: dict):
    # plot avg error cellwise 
    # in case of 3D: plots xy, zy and xz plane in height of hp, zy plane with averaged x_values
    # optional: plots all zy planes, if the condition is set to True

    info = dataloader.dataset.dataset.info

    if len(summed_error_pic.shape)==3:
        extent_highs = (np.array(info["CellsSize"][:3]) * (dataloader.dataset[0][0][0].shape)[-3:])
        summed_error_pics={}

        #zy plane (in height of hp)
        x_position=get_hp_position(summed_error_pic)[0]
        summed_error_pic_zy=remove_first_dim(summed_error_pic, x_position)
        summed_error_pics["zy_slice"]=(summed_error_pic_zy, extent_highs[1:])

        #zy plane with averaged x_values
        summed_error_pic_mean_zy=torch.mean(summed_error_pic,0)
        summed_error_pics["zy_mean"]=(summed_error_pic_mean_zy, extent_highs[1:])

        #plots all zy planes, if set to True
        if False:
            for x_position in range(summed_error_pic.shape[0]):
                summed_error_pic_yz=remove_first_dim(summed_error_pic, x_position)
                summed_error_pics[f"zy_slice_{x_position}"]=summed_error_pic_yz, extent_highs[1:]

        #transpose for xy plane (in height of hp)
        summed_error_pic=summed_error_pic.transpose(0,2)# (X,Y,Z) -> (Z,Y,X)   
        z_position=get_hp_position(summed_error_pic)[0]
        summed_error_pic_xy=remove_first_dim(summed_error_pic, z_position)
        summed_error_pics["xy_slice"]=(summed_error_pic_xy, extent_highs.T[1:])

        #xz plane (in height of hp)
        y_position=get_hp_position(summed_error_pic)[1]
        summed_error_pic=summed_error_pic.transpose(0,1)# (Z,Y,X) -> (Y,Z,X)
        summed_error_pic_xz=remove_first_dim(summed_error_pic, y_position)
        summed_error_pics["xz_slice"]=(summed_error_pic_xz, np.array([extent_highs[2],extent_highs[0]]))

    else:
        extent_highs = (np.array(info["CellsSize"][:2]) * (dataloader.dataset[0][0][0].shape)[-2:])
        summed_error_pics={"xy": (summed_error_pic, extent_highs)}

    for dim, (tensor, ex) in summed_error_pics.items():
        extent = (0,int(ex[0]),int(ex[1]),0) 
        plt.figure()
        plt.imshow(tensor.T, cmap="RdBu_r", extent=extent)
        plt.gca().invert_yaxis()
        plt.ylabel(f"{dim[0]} [m]")
        plt.xlabel(f"{dim[1]} [m]")
        plt.title("Cellwise averaged error in [°C]")
        _aligned_colorbar()

        plt.tight_layout()
        plt.savefig(f"{settings_pic['folder']}/avg_error_{dim}.{settings_pic['format']}", format=settings_pic['format'])
        plt.close()

def _aligned_colorbar(*args, **kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes(
        "right", size=0.3, pad=0.05)
    plt.colorbar(*args, cax=cax, **kwargs)
