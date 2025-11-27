import argparse
import logging
import multiprocessing
import numpy as np
import time
import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss, SmoothL1Loss

from data_stuff.dataset import SimulationDataset, DatasetExtend1, DatasetExtend2, get_splits
from data_stuff.utils import SettingsTraining
from networks.unet import UNet, UNetBC
from networks.unetHalfPad import UNetHalfPad
from networks.unet3d import UNet3d
from processing.solver import Solver
from preprocessing.prepare import prepare_data_and_paths
from postprocessing.visualization import plot_avg_error_cellwise, visualizations, infer_all_and_summed_pic
from postprocessing.measurements import save_all_measurements, measure_additional_losses

def init_data(settings: SettingsTraining, seed=1):
    if settings.problem in ["2stages", "3d"]:
        dataset = SimulationDataset(settings.dataset_prep)
    elif settings.problem == "extend1":
        dataset = DatasetExtend1(settings.dataset_prep, box_size=settings.len_box)
    elif settings.problem == "extend2":
        dataset = DatasetExtend2(settings.dataset_prep, box_size=settings.len_box, skip_per_dir=settings.skip_per_dir)
        settings.inputs += "T"
    print(f"Length of dataset: {len(dataset)}\n")
    generator = torch.Generator().manual_seed(seed)

    split_ratios = [0.7, 0.2, 0.1]
    # if settings.case == "test":
    #     split_ratios = [0.0, 0.0, 1.0] 

    datasets = random_split(dataset, get_splits(len(dataset), split_ratios), generator=generator)
    dataloaders = {}
    if settings.problem == "3d":
        batchsize=6
    else:
        batchsize=1000
    
    try:
        dataloaders["train"] = DataLoader(datasets[0], batch_size=batchsize, shuffle=True, num_workers=0)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=batchsize, shuffle=True, num_workers=0)
    except: pass
    dataloaders["test"] = DataLoader(datasets[2], batch_size=batchsize, shuffle=True, num_workers=0)
    return dataset.input_channels, dataloaders


def run(settings: SettingsTraining):
    multiprocessing.set_start_method("spawn", force=True)
    
    times = {}
    times["time_begin"] = time.perf_counter()
    times["timestamp_begin"] = time.ctime()

    input_channels, dataloaders = init_data(settings)
    # model
    if settings.problem == "2stages":
        model = UNet(in_channels=input_channels).float()
    elif settings.problem in ["extend1", "extend2"]:
        model = UNetHalfPad(in_channels=input_channels).float()
    elif settings.problem == "3d":
        model = UNet3d(in_channels=input_channels).float()
    if settings.case in ["test", "finetune"]:
        model.load(settings.model, settings.device)
    model.to(settings.device)

    solver = None
    if settings.case in ["train", "finetune"]:
        if settings.problem == "3d":
            loss_fn = SmoothL1Loss(beta=0.1)
        else:
            loss_fn = MSELoss()
        # training
        finetune = True if settings.case == "finetune" else False
        solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=loss_fn, finetune=finetune)
        try:
            auto_lr_scheduler = True #set False for own lr_schedule, True for automatic lr_schedule
            if not auto_lr_scheduler:
                solver.load_lr_schedule(settings.destination / "learning_rate_history.csv", settings.case_2hp)
            times["time_initializations"] = time.perf_counter()
            solver.train(settings, auto_lr_scheduler)
            times["time_training"] = time.perf_counter()
        except KeyboardInterrupt:
            times["time_training"] = time.perf_counter()
            logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
        finally:
            solver.save_lr_schedule(settings.destination / "learning_rate_history.csv")
            print("Training finished")

    # save model
    model.save(settings.destination)
    times["time_end"] = time.perf_counter()
    
    which_dataset = "val"
    if settings.case == "test":
        settings.visualize = True
        which_dataset = "test"

    # visualization
    pic_format = "png"
    if settings.visualize:
        start= time.perf_counter()
        visualizations(model, dataloaders[which_dataset], settings.device, plot_path=settings.destination / f"plot_{which_dataset}", amount_datapoints_to_visu=5, pic_format=pic_format)
        times[f"avg_inference_time of {which_dataset}"], summed_error_pic = infer_all_and_summed_pic(model, dataloaders[which_dataset], settings.device)
        plot_avg_error_cellwise(dataloaders[which_dataset], summed_error_pic, {"folder" : settings.destination, "format": pic_format})
        times["duration of visualization in seconds"] = time.perf_counter()-start
        print("Visualizations finished")

    #measurements
    print(f"Start measurements of {which_dataset}")
    start= time.perf_counter()
    times[f"avg_inference_time of {which_dataset}"], summed_error_pic = infer_all_and_summed_pic(model, dataloaders[which_dataset], settings.device)
    errors=measure_additional_losses(model, dataloaders[which_dataset], settings, summed_error_pic)
    times[f"duration of measurement in seconds"]=time.perf_counter()-start
    print("Measurements finished\n")

    save_all_measurements(model, settings, len(dataloaders[which_dataset].dataset), times, solver, errors)
    print(f"Whole process took {(times['time_end']-times['time_begin'])//60} minutes {np.round((times['time_end']-times['time_begin'])%60, 1)} seconds\nOutput in {settings.destination.parent.name}/{settings.destination.name}")

    return model

def save_inference(model_name:str, in_channels: int, settings: SettingsTraining):
    # push all datapoints through and save all outputs
    print("Start inference...\n")
    time_start = time.perf_counter()
    if settings.problem == "2stages":
        model = UNet(in_channels=in_channels).float()
    elif settings.problem in ["extend1", "extend2"]:
        model = UNetHalfPad(in_channels=in_channels).float()
    elif settings.problem == "3d":
        model = UNet3d(in_channels=in_channels).float()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    model.load(model_name, settings.device)
    end.record()
    torch.cuda.synchronize()
    time_model = start.elapsed_time(end)/1000

    model.eval()
    
    data_dir = settings.dataset_prep
    (data_dir / "Outputs").mkdir(exist_ok=True)
    counter=0
    time_load_and_transfer=0
    time_infer=0

    for datapoint in (data_dir / "Inputs").iterdir():
        start.record()
        data = torch.load(datapoint)
        data = torch.unsqueeze(data, 0)
        data=data.to(settings.device)
        end.record()
        torch.cuda.synchronize()
        time_load_and_transfer += start.elapsed_time(end)/1000

        start.record()
        y_out = model(data)
        end.record()
        torch.cuda.synchronize()
        time_infer += start.elapsed_time(end)/1000

        y_out = y_out.detach().cpu()
        y_out = torch.squeeze(y_out, 0)
        torch.save(y_out, data_dir / "Outputs" / datapoint.name)
        counter+=1

    with open(Path.cwd() / "runs" / settings.destination / "measurements_inference.yaml", "w") as f:
        f.write(f"Model loading took {time_model} seconds\n")
        f.write(f"Number of datapoints: {counter}\n")
        f.write(f"Averaged inference time of one datapoint: {time_infer/counter} seconds\n")
        f.write(f"Averaged loading and transfer time of one datapoint: {time_load_and_transfer/counter} seconds\n")
        f.write(f"Whole process of inference took {time.perf_counter() - time_start} seconds\n")
        f.write(f"Outputs saved in {data_dir / 'Outputs'}")
    print(f"Inference finished. Outputs saved in {data_dir / 'Outputs'}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_raw", type=str, default="dataset_2d_small_1000dp", help="Name of the raw dataset (without inputs)")
    parser.add_argument("--dataset_prep", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--case", type=str, choices=["train", "test", "finetune"], default="train")
    parser.add_argument("--model", type=str, default="default") # required for testing or finetuning
    parser.add_argument("--destination", type=str, default="")
    parser.add_argument("--inputs", type=str, default="gksi") #choices=["gki", "gksi", "pksi", "gks", "gksi100", "ogksi1000", "gksi1000", "pksi100", "pksi1000", "ogksi1000_finetune", "gki100", "t", "gkiab", "gksiab", "gkt"]
    parser.add_argument("--case_2hp", type=bool, default=False)
    parser.add_argument("--visualize", type=bool, default=False)
    parser.add_argument("--save_inference", type=bool, default=False)
    parser.add_argument("--problem", type=str, choices=["2stages", "allin1", "extend1", "extend2", "3d",], default="2stages")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--len_box", type=int, default=256)
    parser.add_argument("--skip_per_dir", type=int, default=256)
    args = parser.parse_args()
    settings = SettingsTraining(**vars(args))

    settings = prepare_data_and_paths(settings)

    model = run(settings)

    if args.save_inference:
        save_inference(settings.model, len(args.inputs), settings)