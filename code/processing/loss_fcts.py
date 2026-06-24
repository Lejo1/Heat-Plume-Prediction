import torch.nn as nn
import torch
from torch import max, abs, zeros, sum
from skimage.metrics import structural_similarity as ssim
import yaml
from copy import deepcopy
import contextlib

from preprocessing.transforms import NormalizeTransform
from processing.equations_of_state import eos_water_density_IFC67, eos_water_enthalphy
    
class CombiLoss(nn.Module):
    """
    Loss function that combines MSE and MAE loss with a certain ratio alpha
    """
    def __init__(self, alpha: float = 1., second_loss:nn.Module = nn.L1Loss()):
        super(CombiLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.secondary_loss_function = second_loss
        self.alpha = alpha
        self.name = rf"CombiLoss (a={alpha}) with {self.secondary_loss_function}"

    def forward(self, predictions, labels):
        eval_second = self.secondary_loss_function(predictions, labels)

        return self.alpha * self.mse(predictions, labels) + (1. - self.alpha) * eval_second
    

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, predictions, labels):
        num_channels = predictions.shape[1]
        ssim_total = 0
        for dp in range(predictions.shape[0]):
            for channel in range(num_channels):
                pred = predictions[dp, channel].detach().cpu().numpy()
                lab  = labels[dp, channel].detach().cpu().numpy()

                data_range = max(pred.max(), lab.max()) - min(pred.min(), lab.min())
                ssim_total += ssim(pred, lab, data_range=data_range)

        return ssim_total / num_channels / predictions.shape[0]
    

class LinfLoss(nn.Module):
    def __init__(self):
        super(LinfLoss, self).__init__()

    def forward(self, output, target):
        return max(abs(output - target))


class PATLoss(nn.Module):
    """
    Percentage above Threshold, unit [%]
    pat = torch.sum(torch.abs(y_pred[:,0] - y[:,0]) > pbt_thresholds[idx])
    """

    def __init__(self, pat_thresholds: list):
        super(PATLoss, self).__init__()
        self.pat_thresholds = pat_thresholds

    def forward(self, output, label):
        if len(output.shape) == 3:
            output = output.unsqueeze(1)
            label = label.unsqueeze(1)
        pat = zeros((output.shape[0], len(self.pat_thresholds)), device=output.device)
        for idx in range(output.shape[1]):
            pat[:, idx] = sum(abs(output[:, idx] - label[:, idx]) > self.pat_thresholds[idx], dim=(1, 2)) / (output.shape[2] * output.shape[3])
        return pat * 100
    
class EnergyLoss(nn.Module):
    def __init__(self, data_dir, device:str="cuda:0", data_type=torch.float32, half_precision=False, keep_dim:bool=False):
        super(EnergyLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.norm_info = yaml.load(open(data_dir+"info.yaml"), Loader=yaml.SafeLoader)
        assert "Liquid X-Velocity [m_per_y]" in self.norm_info["Inputs"], "Velocity-x not in Inputs"
        self.norm = NormalizeTransform(self.norm_info)
        self.device = device
        self.data_type = data_type
        self.half_precision = half_precision
        self.kernel = torch.tensor([[-1,0,1],
                                    [0,0,0],
                                    [1,0,-1]],dtype=data_type, device=self.device).unsqueeze(0).unsqueeze(0)
        self.keep_dim = keep_dim
        self.norm_factor = 1e-16

    def forward(self, prediction, inputs):
        inputs_full = deepcopy(inputs[0].detach()).to(torch.float32)
        self.inputs_unnormed = self.norm.reverse(deepcopy(inputs_full), "Inputs")
        self.pressure = self.inputs_unnormed[self.norm_info["Inputs"]["Liquid Pressure [Pa]"]["index"]].to(self.device)
        self.vx = self.inputs_unnormed[self.norm_info["Inputs"]["Liquid X-Velocity [m_per_y]"]["index"]].to(self.device)
        self.vy = self.inputs_unnormed[self.norm_info["Inputs"]["Liquid Y-Velocity [m_per_y]"]["index"]].to(self.device)
        self.ids_normed = inputs[0][:,self.norm_info["Inputs"]["Material ID"]["index"]].to(self.device)
        predicted_T_unnormed = self.norm.reverse(deepcopy(prediction.detach().to(torch.float32).requires_grad_(True)), "Labels").squeeze()
        # TODO dimensions! expect 2D
        loss = energy_loss(self.pressure, predicted_T_unnormed, self.vx, self.vy, self.ids_normed, self.mse_loss, self.kernel, data_type=self.data_type, half_precision=self.half_precision, device=self.device, keep_dim=self.keep_dim)
        return loss * self.norm_factor

def energy_loss(pressure, predicted_temperature, vx, vy, ids_normed, mse_loss, kernel, data_type=torch.float32, half_precision=False, device:str='cuda:0', keep_dim:bool=False):
    #  based on : Manuel Hirche, Bachelor thesis, 2023
    resolution = 5. #m
    # cond_dry = 0.65
    # cond_sat = 1.0
    # sl  = 1 #? saturation of liquid?
    thermal_conductivity = 1 #cond_dry + torch.sqrt(sl) * (cond_sat - cond_dry)
    # Calculate density, molar_density, and enthalpy
    density, molar_density = eos_water_density_IFC67(predicted_temperature, pressure)
    enthalpy = eos_water_enthalphy(predicted_temperature, pressure)
    # Calculate temperature gradients
    T_grad = torch.gradient(predicted_temperature, dim=(1,2))
    # Calculate energy components
    energy_u = torch.gradient((molar_density * vx * enthalpy) - (thermal_conductivity * T_grad[0]/resolution), dim=(1,2))[0]/resolution
    energy_v = torch.gradient((molar_density * vy * enthalpy) - (thermal_conductivity * T_grad[1]/resolution), dim=(1,2))[1]/resolution
    energy = energy_u + energy_v

    # Calculate inflow energy
    inflow_energy = energy_hps(ids_normed, resolution, density, kernel, data_type=data_type, half_precision=half_precision, device=device)
    energy -= inflow_energy #*0.5

    # Calculate energy loss
    if not keep_dim:
        energy_loss = mse_loss(energy, torch.zeros_like(energy))
    else:
        energy_loss = torch.nn.MSELoss(reduction="none")(energy, torch.zeros_like(energy))
    return energy_loss

def energy_hps(ids, resolution, density, kernel, data_type=torch.float32, half_precision=False, device:str='cuda:0'):

    with (torch.autocast(device_type=device, dtype=data_type) if half_precision else contextlib.nullcontext()):
        specific_heat_water = 4200 # [J/kgK]
        density_water = density # [kg/m^3]
        temp_diff = 5 # [K]
        volumetric_flow_rate = 0.00024 # [m^3/s]

        hp_energy = specific_heat_water * density_water * temp_diff * volumetric_flow_rate * 1/resolution**3
        hp_energy = hp_energy * ids
        
        hp_energy = torch.nn.functional.conv2d(hp_energy.unsqueeze(1), kernel, padding=1)

    return (hp_energy[:,0])