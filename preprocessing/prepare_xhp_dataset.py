import logging
import pathlib
import time
from typing import List

import yaml
from torch import equal, mean
from tqdm.auto import tqdm

from preprocessing.domain_classes.domain import Domain
from preprocessing.domain_classes.heat_pump import HeatPumpBox
from preprocessing.domain_classes.utils_2hp import (
    save_config_of_separate_inputs)
from preprocessing.prepare_1ststage import prepare_dataset
from preprocessing.prepare_paths import Paths2HP
from networks.unet import UNet
from networks.unetQuad import UNetQuad
from networks.unetParallel import UNetParallel
from data_stuff.utils import SettingsTraining
from copy import deepcopy
from postprocessing.visualization import plot_datafields, DataToVisualize


def prepare_xhp_dataset(paths: Paths2HP, settings:SettingsTraining):
    """
    prepare dataset for a domain with x heat pumps
    for good quality the model should be trained on x-1 heat pumps
    """
    
    timestamp_begin = time.ctime()
    time_begin = time.perf_counter()

    ## load model
    time_start_prep_domain = time.perf_counter()
    if settings.architecture == "standard":
        model_1HP = UNet(in_channels=len(settings.inputs)).float()
    elif settings.architecture == "quad":
        model_1HP = UNetQuad(in_channels=len(settings.inputs)).float()
    elif settings.architecture == "parallel":
        model_1HP = UNetParallel(in_channels=len(settings.inputs)).float()
    model_1HP.load(paths.model_1hp_path, map_location=settings.device)
    model_1HP.eval()
    
    # prepare domain dataset   
    # norm with data from dataset that NN was trained with!!
    with open(paths.dataset_model_trained_with_prep_path / "info.yaml", "r") as file:
        info = yaml.safe_load(file)
    prepare_dataset(paths, settings.inputs, info=info, power2trafo=False) #for using unet on whole domain required: power2trafo=True
    print(f"Domain prepared ({paths.dataset_1st_prep_path})")

    # prepare dataset
    time_start_prep_2hp = time.perf_counter()
    avg_time_inference_1hp = 0
    list_runs = list(pathlib.Path(paths.dataset_1st_prep_path / "Inputs").iterdir())
    for run_path in tqdm(list_runs, desc="2HP prepare", total=len(list_runs)):
        # for each run, load domain and 1hp-boxes
        run_file = str(run_path).split("/")[-1]
        run_id = f'{run_file.split(".")[0]}_'
        domain = Domain(paths.dataset_1st_prep_path, stitching_method="max", file_name=run_file)
        if mean(domain.prediction) > 10:
            domain.prediction = domain.norm(domain.prediction.clone().detach(), property="Temperature [C]")

        if domain.skip_datapoint:
            logging.warning(f"Skipping {run_id}")
            continue

        ## generate 1hp-boxes and extract information like perm and ids etc.
        single_hps = domain.extract_hp_boxes(settings.device)
        # generate dataset
        single_hps, avg_time_inference_1hp = prepare_hp_boxes(paths, model_1HP, single_hps, domain, run_id,settings, avg_time_inference_1hp, save_bool=True)
        
    time_end = time.perf_counter()
    avg_inference_times = avg_time_inference_1hp / len(list_runs)
    print(f"Datasets saved in {paths.datasets_boxes_prep_path}")
    # save infos of info file about separated (only 2!) inputs
    save_config_of_separate_inputs(domain.info, path=paths.datasets_boxes_prep_path)

    # save measurements
    with open(paths.datasets_boxes_prep_path / "measurements.yaml", "w") as f:
        f.write(f"timestamp of beginning: {timestamp_begin}\n")
        f.write(f"timestamp of end: {time.ctime()}\n")
        f.write(f"model 1HP: {paths.model_1hp_path}\n")
        f.write(f"input params: {settings.inputs}\n")
        f.write(f"separate inputs: {True}\n")
        f.write(f"location of prepared domain dataset: {paths.dataset_1st_prep_path}\n")
        f.write(f"name of dataset prepared with: {paths.dataset_model_trained_with_prep_path}\n")
        f.write(f"name of dataset domain: {paths.raw_path.name}\n")
        f.write(f"name_destination_folder: {paths.datasets_boxes_prep_path}\n")
        f.write(f"avg inference times for 1HP-NN in seconds: {avg_inference_times}\n")
        f.write(f"device: {settings.device}\n")
        f.write(f"duration of preparing domain in seconds: {(time_start_prep_2hp-time_start_prep_domain)}\n")
        f.write(f"duration of preparing 2HP in seconds: {(time_end-time_start_prep_2hp)}\n")
        f.write(f"duration of preparing 2HP /run in seconds: {(time_end-time_start_prep_2hp)/len(list_runs)}\n")
        f.write(f"duration of whole process in seconds: {(time_end-time_begin)}\n")

    return domain, single_hps

def prepare_hp_boxes(paths:Paths2HP, model_1HP:UNet, single_hps:List[HeatPumpBox], domain:Domain, run_id:int,settings:SettingsTraining, avg_time_inference_1hp:float=0, save_bool:bool=True):
    hp: HeatPumpBox
    """
    prepare the dataset by predicting the temperature field around every heat pump
    starts by generating a set of permutations for the order in which the temperature field around a heat pump is predicted
    every heat pump is last once, as we assume the step from x-1 heat pump to x heat pump to be the most important
    saves them too

    Assumes the temperature field to be the last input parameter!!!
    """

    #heat pumps identified through position in domain
    positions = []
    for hp in single_hps:
        positions.append(hp.pos)
    
    settings_pic = {"format": "png",
        "dpi": 600,} 

    # generate permutations (not all of them, ensure that every heat pump is last once) 
    relevant_permutations = []
    for index in range(len(positions)):
        relevant_permutations.append(deepcopy(positions))
        positions[index], positions[-1] = positions[-1], positions[index]

    current_permutation = 0
    # iterate through heat pumps in order of the permutation
    for permutation in relevant_permutations:
        count = 0
        # deepcopy to ensure we dont have to reset heat pumps after every permutation
        hps_copy = deepcopy(single_hps)
        for hp in hps_copy:
            hp.primary_temp_field = domain.norm(hp.primary_temp_field, property="Temperature [C]")
            hp.other_temp_field = domain.norm(hp.other_temp_field, property="Temperature [C]")

        for position in permutation:
            for hp in hps_copy:
                if equal(hp.pos,position):
                    current_hp = hp
            
            # get input temperature field except for first heat pump, need temperature field to be the last input!!!
            if count > 0:
                current_hp.get_other_temp_field(hps_copy)
                current_hp.inputs[-1] = current_hp.other_temp_field.clone().detach()
            
            # apply cnn
            time_start_run_1hp = time.perf_counter()
            current_hp.primary_temp_field = current_hp.apply_nn(model_1HP)
            avg_time_inference_1hp += time.perf_counter() - time_start_run_1hp

            # plot datapoints and save them
            dict_to_plot = {}
            pathlib.Path(settings.destination / run_id).mkdir(parents=True, exist_ok=True)
            name_id = "perm_" + str(current_permutation) + "_hp_" +str(count)
            name_pic = settings.destination / run_id / name_id

            dict_to_plot[f"input{run_id}{count}"] = DataToVisualize(domain.reverse_norm(current_hp.inputs[-1].clone().detach(), property="Temperature [C]"), f"input {run_id}_{count}")

            if count == len(permutation)-1:
                dict_to_plot[f"label{run_id}{count}"] = DataToVisualize(domain.reverse_norm(current_hp.label.clone().detach().squeeze(), property="Temperature [C]"), f"label {run_id}_{count}")
                current_hp.save(run_id=run_id+str(current_permutation), dir=paths.datasets_boxes_prep_path/f"{count+1}HP",)
            else:
                dict_to_plot[f"label{run_id}{count}"] = DataToVisualize(domain.reverse_norm(current_hp.primary_temp_field.clone().detach().squeeze(), property="Temperature [C]"), f"label {run_id}_{count}")
                current_hp.save(run_id=run_id+str(current_permutation), dir=paths.datasets_boxes_prep_path/f"{count+1}HP", alt_label=current_hp.primary_temp_field.clone().detach(),)
            count = count + 1
            plot_datafields(dict_to_plot, name_pic, settings_pic)
        current_permutation = current_permutation + 1 
    return single_hps, avg_time_inference_1hp