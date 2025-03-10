import argparse
import logging
import multiprocessing
import numpy as np
import time
import torch
from torch.nn import MSELoss

from data_stuff.utils import SettingsTraining
from data_stuff.init_data import init_data
from networks.unet import UNet
from networks.unetQuad import UNetQuad
from networks.unetParallel import UNetParallel
from processing.solver import Solver
from processing.hypertune import tune_nn
from preprocessing.prepare import prepare_data_and_paths
from postprocessing.visualization import plot_avg_error_cellwise, visualizations, infer_all_and_summed_pic, visualize_dataset
from postprocessing.measurements import measure_loss, save_all_measurements
from postprocessing.iterative_estimation import iterative_estimation
from torchsummary import summary
from preprocessing.prepare_paths import Paths2HP

def run(settings: SettingsTraining, paths: Paths2HP):
    multiprocessing.set_start_method("spawn", force=True)

    #hyperparamter tuning, no training and visualization
    if settings.case in ["hypertune"]:
        tune_nn(settings)
        return
    
    times = {}
    times["time_begin"] = time.perf_counter()
    times["timestamp_begin"] = time.ctime()

    input_channels, dataloaders = init_data(settings)
    # model
    if settings.architecture == "standard":
        model = UNet(in_channels=input_channels).float()
    elif settings.architecture == "parallel":
        model = UNetParallel(in_channels=input_channels).float()
    elif settings.architecture == "quad":
        model = UNetQuad(in_channels=input_channels).float()

    if settings.case in ["test", "finetune","iterative"]:
        model.load(settings.model, map_location=settings.device)
    
    #only visualization for evaluation, skip other stuff
    if settings.case == "visualize":
        visualize_dataset(dataloaders["test"], settings.device, plot_path=settings.destination / f"plot_vis", amount_datapoints_to_visu=20, pic_format="png")
        return

    #use model for iterative estimation of whole domain 
    if settings.case == "iterative":
        model.to("cpu")
        model = iterative_estimation(model,settings, paths)
        return model
    
    model.to(settings.device)
    solver = None
    if settings.case in ["train", "finetune"]:
        loss_fn = MSELoss()
        # training
        finetune = True if settings.case == "finetune" else False
        solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=loss_fn, finetune=finetune)
        try:
            solver.load_lr_schedule(settings.destination / "learning_rate_history.csv", False)
            times["time_initializations"] = time.perf_counter()
            solver.train(settings)
            times["time_training"] = time.perf_counter()
        except KeyboardInterrupt:
            times["time_training"] = time.perf_counter()
            logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
        finally:
            solver.save_lr_schedule(settings.destination / "learning_rate_history.csv")
            print("Training finished")

    # save model
    model.save(settings.destination)
    summary(model,input_size=(5,64,256))

    # visualization
    which_dataset = "val"
    pic_format = "png"
    times["time_end"] = time.perf_counter()
    errors = {}
    if settings.case == "test":
        settings.visualize = True
        which_dataset = "test"
        errors = measure_loss(model, dataloaders, settings.device, vT_case="temperature")
        # errors = measure_loss(model, dataloaders[which_dataset], settings.device)
    if settings.visualize:
        errors["isolines"] = visualizations(model, dataloaders[which_dataset], settings.device, plot_path=settings.destination / f"plot_{which_dataset}", amount_datapoints_to_visu=10, pic_format=pic_format)
        times[f"avg_inference_time of {which_dataset}"], summed_error_pic = infer_all_and_summed_pic(model, dataloaders[which_dataset], settings.device)
        plot_avg_error_cellwise(dataloaders[which_dataset], summed_error_pic, {"folder" : settings.destination, "format": pic_format})
        print("Visualizations finished")
    save_all_measurements(settings, len(dataloaders[which_dataset].dataset), times, solver, errors)       
    print(f"Whole process took {(times['time_end']-times['time_begin'])//60} minutes {np.round((times['time_end']-times['time_begin'])%60, 1)} seconds\nOutput in {settings.destination.parent.name}/{settings.destination.name}")

    return model

def save_inference(model_name:str, in_channels: int, settings: SettingsTraining):
    # push all datapoints through and save all outputs
    if settings.architecture == "standard":
        model = UNet(in_channels=in_channels).float()

    model.load(model_name, map_location=settings.device)
    model.eval()

    data_dir = settings.dataset_prep
    (data_dir / "Outputs").mkdir(exist_ok=True)

    for datapoint in (data_dir / "Inputs").iterdir():
        data = torch.load(datapoint)
        data = torch.unsqueeze(data, 0)
        time_start = time.perf_counter()
        y_out = model(data.to(settings.device)).to(settings.device)
        time_end = time.perf_counter()
        y_out = y_out.detach().cpu()
        y_out = torch.squeeze(y_out, 0)
        torch.save(y_out, data_dir / "Outputs" / datapoint.name)
        print(f"Inference of {datapoint.name} took {time_end-time_start} seconds")
    
    print(f"Inference finished, outputs saved in {data_dir / 'Outputs'}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_raw", type=str, default="dataset_2hps_1fixed_1000dp", help="Name of the raw dataset (without inputs)")
    parser.add_argument("--dataset_prep", type=str, default="", help="Name of the prepared dataset")
    parser.add_argument("--device", type=str, default="cuda:0", help="device for torch (cpu, gpu)")
    parser.add_argument("--epochs", type=int, default=10000, help="For how many epochs the network should be trained")
    parser.add_argument("--case", type=str, choices=["train", "test", "finetune", "hypertune", "visualize", "iterative", "prepare"], default="train", help="case of the current execution, eg train, test, hyperparameter tuning (hypertune)...")
    parser.add_argument("--model", type=str, default="default", help="Name of the model")
    parser.add_argument("--destination", type=str, default="default_dest", help="destination folder name")
    parser.add_argument("--inputs", type=str, default="gksit", help="input parameters")
    parser.add_argument("--visualize", type=bool, default=False, help="Flag for visualizing result")
    parser.add_argument("--already_prep", type=bool, default=False, help="Flag when only prepared dataset is available")
    parser.add_argument("--save_inference", type=bool, default=False, help="Flag for saving measurements")
    parser.add_argument("--architecture", type=str, choices=["standard","parallel","quad"], default="standard", help="Architecture of model")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--len_box", type=int, default=256)
    parser.add_argument("--skip_per_dir", type=int, default=256)
    args = parser.parse_args()
    settings = SettingsTraining(**vars(args))

    if settings.model == "default" and settings.case in ["test", "finetune", "visualize", "iterative"]:
        print(f"for case {settings.case} a model is required!")
    else:
        settings, paths = prepare_data_and_paths(settings)
        if not settings.case == "prepare":
            model = run(settings, paths)
            if args.save_inference:
                save_inference(settings.model, len(args.inputs), settings)