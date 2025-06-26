from dataclasses import dataclass, field
from math import inf
from typing import Dict
# set backend for matplotlib, necessary if no GUI is available, e.g. in tmux
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from preprocessing.transforms import NormalizeTransform
from processing.networks.unetVariants import UNet
import postprocessing.cmap_jp

@dataclass
class DataToVisualize:
    data: np.ndarray
    category: str
    physical_property: str
    extent_highs :tuple = (1280,100) # x,y in meters
    imshowargs: Dict = field(default_factory=dict)
    contourfargs: Dict = field(default_factory=dict)
    contourargs: Dict = field(default_factory=dict)
    vmax: float = None
    vmin: float = None

    def __post_init__(self):
        extent = (0,int(self.extent_highs[0]),int(self.extent_highs[1]),0)

        if "temperature" in self.physical_property.lower():
            cmap = "jp_temperature"
        elif "material" in self.physical_property.lower():
            cmap = "binary"
        else:
            cmap = "jp_linear"
            
        self.imshowargs = {"cmap": cmap, 
                           "extent": extent,
                           "interpolation": "nearest",}
        if self.vmax is not None:
            self.imshowargs["vmax"] = self.vmax
        if self.vmin is not None:
            self.imshowargs["vmin"] = self.vmin

        self.contourfargs = {"levels": np.arange(10.4, 16, 0.25), 
                             "cmap": cmap, 
                             "extent": extent}
        
        T_gwf = 10.6
        self.contourargs = {"levels" : [np.round(T_gwf + 1, 1)],
                            "cmap" : "Pastel1", 
                            "extent": extent}

        if self.physical_property == "Liquid Pressure [Pa]":
            self.physical_property = "Pressure [Pa]"
        elif self.physical_property == "Material ID":
            self.physical_property = "Positions of Heat Pumps [-]"
        elif self.physical_property == "Permeability X [m^2]":
            self.physical_property = "Permeability [m$^2$]"
        elif self.physical_property == "SDF":
            self.physical_property = "SDF-Transformed Positions of Heat Pumps [-]"
        elif self.physical_property == "MDF":
            self.physical_property = "MDF-Transformed Positions of Heat Pumps [-]"
        elif self.physical_property == "Streamlines Fade":
            self.physical_property = "Streamlines Fade [-]"
        elif self.physical_property == "Streamlines":
            self.physical_property = "Streamlines [-]"

def visualizations(model: UNet, dataloader: DataLoader, args: dict, amount_datapoints_to_visu: int = inf, plot_path: str = "default", pic_format: str = "png"):
    print("Visualizing...") #, end="\r")

    if amount_datapoints_to_visu > len(dataloader.dataset):
        amount_datapoints_to_visu = len(dataloader.dataset)

    try:
        norm = dataloader.dataset.norm
        info = dataloader.dataset.info
    except AttributeError:
        norm = dataloader.dataset.dataset.norm
        info = dataloader.dataset.dataset.info
    settings_pic = {"format": pic_format,
                    "dpi": 1200,}
    
    current_id = 0
    for inputs, labels in dataloader:
        print(inputs.shape, labels.shape, "shape of inputs and labels")
        len_batch = inputs.shape[0]
        for datapoint_id in range(len_batch):
            name_pic = f"{plot_path}_{current_id}"

            x = inputs[datapoint_id]
            y = labels[datapoint_id]

            y_out = model.infer(x.unsqueeze(0), args["device"])

            x, y, y_out = reverse_norm_one_dp(x, y, y_out, norm)
            dict_to_plot = prepare_data_to_plot(x, y, y_out, info)

            plot_datafields(dict_to_plot, name_pic, settings_pic, only_inner=False, plot_all_in_1_pic=False)

            if current_id >= amount_datapoints_to_visu-1:
                return None
            current_id += 1

def reverse_norm_one_dp(x: torch.Tensor, y: torch.Tensor, y_out:torch.Tensor, norm: NormalizeTransform):
    # reverse transform for plotting real values
    x = norm.reverse(x.detach().cpu().squeeze(0), "Inputs")
    if len(y.shape) == 4:
        y = norm.reverse(y.detach().cpu().squeeze(0),"Labels")
    else:
        y = norm.reverse(y.detach().cpu(),"Labels")
    try:
        y_out = norm.reverse(y_out.detach().cpu().squeeze(0),"Labels")
    except:
        y_out = norm.reverse(y_out.squeeze(0),"Labels")
    return x, y, y_out

def prepare_data_to_plot(x: torch.Tensor, y: torch.Tensor, y_out:torch.Tensor, info: dict):
    # prepare data of temperature true, temperature out, error, physical variables (inputs)
    required_size = y_out.shape
    start_pos = ((y.shape[1] - required_size[1])//2, (y.shape[2] - required_size[2])//2)
    y_reduced = y[:,start_pos[0]:start_pos[0]+required_size[1], start_pos[1]:start_pos[1]+required_size[2]]
    
    outs_max = [max(y_reduced[idx].max(), y_out[idx].max()) for idx in range(len(y_reduced))]
    outs_min = [min(y_reduced[idx].min(), y_out[idx].min()) for idx in range(len(y_reduced))]
    extent_highs_y = (np.array(info["CellsSize"][:2]) * y_out.shape[-2:])
    
    dict_to_plot = {}
    labels = info["Labels"].keys()
    for label in labels:
        index = info["Labels"][label]["index"]
        dict_to_plot[f"{label}_true"] = DataToVisualize(y_reduced[index], "Label", label, extent_highs_y, vmax=outs_max[index], vmin=outs_min[index])
        dict_to_plot[f"{label}_out"] = DataToVisualize(y_out[index], "Prediction", label, extent_highs_y, vmax=outs_max[index], vmin=outs_min[index])
        dict_to_plot[f"{label}_error"] = DataToVisualize(torch.abs(y_reduced[index]-y_out[index]), "Absolute Error", label, extent_highs_y)
    inputs = info["Inputs"].keys()
    for input in inputs:
        index = info["Inputs"][input]["index"]
        dict_to_plot[input] = DataToVisualize(x[index], "Input", input, (np.array(info["CellsSize"][:2]) * x.shape[-2:]))

    return dict_to_plot

def plot_datafields(data: Dict[str, DataToVisualize], name_pic: str, settings_pic: dict, only_inner: bool = False, plot_all_in_1_pic: bool = True):
    # plot datafields (temperature true, temperature out, error, physical variables (inputs))

    if plot_all_in_1_pic:
        num_subplots = len(data)
        fig, axes = plt.subplots(num_subplots, 1, sharex=True)
        fig.set_figheight(num_subplots*3)
        
        for index, (name, datapoint) in enumerate(data.items()):
            plt.sca(axes[index])
            plt.title(datapoint.category)
            if only_inner:
                plt.imshow(datapoint.data[100:400,100:400], **datapoint.imshowargs)
            else:  
                plt.imshow(datapoint.data, **datapoint.imshowargs)
            plt.gca().invert_yaxis()

            plt.ylabel("x [m]")
            aligned_colorbar(label=datapoint.physical_property)

        plt.xlabel("y [m]")
        plt.tight_layout()
        ext_inner = "_inner" if only_inner else ""
        plt.savefig(f"{name_pic}{ext_inner}.{settings_pic['format']}", **settings_pic)
    else:
        for (name, datapoint) in data.items():
            fig, _ = plt.subplots(1, 1, sharex=True)
            fig.set_figheight(5)
            plt.title(datapoint.category)
            if only_inner:
                plt.imshow(datapoint.data[100:400,100:400], **datapoint.imshowargs)
            else:  
                plt.imshow(datapoint.data, **datapoint.imshowargs)

            plt.ylabel("x [m]")
            plt.xlabel("y [m]")
            aligned_colorbar(label=datapoint.physical_property)
            plt.tight_layout()
            ext_inner = "_inner" if only_inner else ""
            plt.savefig(f"{name_pic}{ext_inner}_{name}.{settings_pic['format']}", **settings_pic)


def aligned_colorbar(*args, **kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes(
        "right", size=0.3, pad=0.05)
    plt.colorbar(*args, cax=cax, **kwargs)
