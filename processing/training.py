import logging
import multiprocessing
import numpy as np
import torch
from torch.nn import MSELoss, L1Loss, HuberLoss
from datetime import datetime

from preprocessing.preprocessing import preprocessing as data_prep
from preprocessing.data_init import init_data, load_all_datasets_in_full
from processing.networks.unet import UNet
from processing.networks.unetVariants import UNetHalfPad2, UNetNoPad2
from processing.solver import Solver
from processing.losses import CombiLoss
from postprocessing.visualization import visualizations
from postprocessing.measurements import measure_losses_paper24
from utils.utils_args import save_yaml, load_yaml

def train_old(args: dict):
    np.random.seed(1)
    torch.manual_seed(1)
    multiprocessing.set_start_method("spawn", force=True)

    batchsize=8
    tmp_bool_cutouts = False
    ORDER_DATA = [0,2,1] # in RUN-ID labels this is [0,3,2]
    
    data_prep(args) # and save info.yaml in model folder

    input_channels, output_channels, dataloaders = init_data(args, batchsize=batchsize, tmp_bool_cutouts=tmp_bool_cutouts, ORDER_DATA=ORDER_DATA)
    if output_channels == 1:
        vT_case = "temperature"
    elif output_channels == 2:
        vT_case = "velocities"

    # model
    if args["problem"] in ["1hp", "2stages", "test"]:
        model = UNet(in_channels=input_channels).float()
    elif args["problem"] in ["extend"]:
        model = UNetHalfPad2(in_channels=input_channels).float()
    elif args["problem"] in ["allin1"]:
        if vT_case == "temperature":
            kernel_size = 4 # best setting from optimization with optuna
        elif vT_case == "velocities":
            kernel_size=5 # best setting from optimization with wandb for real
        depth=5
        init_features=16
        model = UNetNoPad2(in_channels=input_channels, out_channels=output_channels, depth=depth, init_features=init_features, kernel_size=kernel_size).float() # best setting from optimization with optuna
    model.to(args["device"])
    
    if args["case"] in ["test", "finetune"]:
        model.load(args["model"], args["device"])
    if args["case"] == "test":
        model.eval()

    if args["case"] in ["train", "finetune"]:
        if vT_case == "temperature":
            loss = L1Loss() #CombiLoss(0.75) #
        elif vT_case == "velocities":
            loss = MSELoss() #CombiLoss(0.75) # MSELoss() # best setting from optimization with optuna
        solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=loss, finetune=(args["case"] == "finetune"))
        training_time = datetime.now()
        try:
            solver.load_lr_schedule(args["destination"] / "learning_rate_history.csv")
            solver.train(args)
        except KeyboardInterrupt:
            logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
        finally:
            solver.save_lr_schedule(args["destination"] / "learning_rate_history.csv")
            print("Training finished")

        # save model and train metrics
        training_time = datetime.now() - training_time
        model.save(args["destination"])
        solver.save_metrics_to_overall_csv(args["destination"], model.num_of_params(), args["epochs"], training_time, args["device"])
        solver.save_metrics_separate_yaml(args["destination"], model.num_of_params(), args["epochs"], training_time.total_seconds(), args["device"])

    # postprocessing
    # save_all_measurements(args, len(dataloaders["val"].dataset), times={}, solver=solver) #, errors)
    measure_losses_paper24(model, dataloaders, args, vT_case=vT_case, tmp_bool_cutouts=tmp_bool_cutouts)

    # if tmp_bool_cutouts:
    #     dataloaders = load_all_datasets_in_full(args, ORDER_DATA=ORDER_DATA)
    # for case in ["train", "val", "test"]:
    case = "train" #args["case"]
    visualizations(model, dataloaders[case], args, plot_path=args["destination"] / case, amount_datapoints_to_visu=1, pic_format="png")

    return model


def train_read_hyperparams(args: dict):
    # np.random.seed(1)
    # torch.manual_seed(1)
    multiprocessing.set_start_method("spawn", force=True)

    args = load_hyperparams(args)
    data_prep(args) # and save info.yaml in model folder
    
    input_channels, output_channels, dataloaders = init_data(args,)

    model = init_model(args, input_channels, output_channels)

    if args["case"] in ["train", "finetune"]:
        loss = init_loss_function(args)
        # TODO change val-loss in solver (realK: Huber, dummyK:mae)
        solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=loss, finetune=(args["case"] == "finetune"), optimizer_switch=args["optimizer_switch"], learning_rate=float(args["lr"]))
        training_time = datetime.now()
        try:
            solver.load_lr_schedule(args["destination"] / "learning_rate_history.csv")
            solver.train(args)
        except KeyboardInterrupt:
            logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
        finally:
            solver.save_lr_schedule(args["destination"] / "learning_rate_history.csv")
            print("Training finished")

        # save model and train metrics
        training_time = datetime.now() - training_time
        model.save(args["destination"])
        # TODO solver.save_metrics_to_overall_csv(args["destination"], model.num_of_params(), args["epochs"], training_time, args["device"])
        solver.save_metrics_separate_yaml(args["destination"], model.num_of_params(), args["epochs"], training_time.total_seconds(), args["device"])

    # postprocessing
    # save_all_measurements(args, len(dataloaders["val"].dataset), times={}, solver=solver) #, errors)
    measure_losses_paper24(model, dataloaders, args, output_channels, tmp_bool_cutouts=args["bool_cutouts"])

    case = "val"
    visualizations(model, dataloaders[case], args, plot_path=args["destination"] / case, amount_datapoints_to_visu=1, pic_format="png")

    return model

def init_model(args, input_channels, output_channels):
    if args["problem"] in ["allin1"]:
        if input_channels == 3 and output_channels == 1: # end-to-end, want to use same data-split
            model = UNet(in_channels=input_channels, out_channels=output_channels, depth=args["depth"], init_features=args["init_features"], kernel_size=args["kernel_size"]).float()
        else:
            model = UNetNoPad2(in_channels=input_channels, out_channels=output_channels, depth=args["depth"], init_features=args["init_features"], kernel_size=args["kernel_size"], stride=args["stride"], dilation=args["dilation"], activation=args["activation_fct"], norm=args["norm"], repeat_inner=args["repeat_inner"]).float()
    elif args["problem"] in ["1hp", "2stages", "test"]:
        model = UNet(in_channels=input_channels, out_channels=output_channels, depth=args["depth"], init_features=args["init_features"], kernel_size=args["kernel_size"]).float()
    elif args["problem"] in ["extend"]:
        model = UNetHalfPad2(in_channels=input_channels, out_channels=output_channels, depth=args["depth"], init_features=args["init_features"], kernel_size=args["kernel_size"], activation=args["activation_fct"], norm=args["norm"]).float()
    model.to(args["device"])
        
    if args["case"] in ["test", "finetune"]:
        model.load(args["model"], args["device"])
    if args["case"] == "test":
        model.eval()
    return model

def init_loss_function(args):
    if args["train_loss"].lower() == "mae":
        loss = L1Loss()
    elif args["train_loss"].lower() == "mse":
        loss = MSELoss()
    elif args["train_loss"].lower() == "huber":
        loss = HuberLoss()
    elif args["train_loss"].lower() == "combi":
        loss = CombiLoss(0.75)
    return loss

def load_hyperparams(args):
    hyperparams = load_yaml(args["destination"] / "HPS_options.yaml")
    for key in hyperparams.keys():
        args[key] = hyperparams[key]["values"][0]
    return args