import optuna
from optuna.trial import TrialState
import argparse
import os
from pathlib import Path
import torch
import sys
import logging
import multiprocessing
import numpy as np
import torch
from torch.nn import MSELoss, L1Loss
from datetime import datetime

from preprocessing.data_init import init_data
from processing.networks.unetVariants import UNetNoPad2
from processing.networks.unet import UNet
from processing.solver import Solver
from postprocessing.visualization import visualizations
from postprocessing.measurements import measure_losses_paper24
from utils.utils_args import save_yaml
import utils.utils_args as ut
import preprocessing.preprocessing as prep
from main import read_cla
from processing.training import train_read_hyperparams

def run(trial):
    config = ut.load_yaml(args["destination"] / "HPS_options.yaml")
    run_name = trial.number
    for key in config:
        args[key] = trial.suggest_categorical(key, config[key]["values"])

    # np.random.seed(1)
    # torch.manual_seed(1)
    multiprocessing.set_start_method("spawn", force=True)

    # args["data_prep"] = f"/scratch/sgs/pelzerja/datasets_prepared/allin1/before_2025/{args['data_raw'].name} inputs_{args['inputs']} outputs_{'xy' if len(args['inputs']) == 2 else 't'}"
    # print(args["data_prep"])

    ut.make_paths(args) # and check if data / model exists
    ut.save_yaml(args, args["destination"] / "command_line_arguments.yaml")

    # data
    prep.preprocessing(args) # and save info.yaml in model folder
    input_channels, output_channels, dataloaders = init_data(args, tmp_bool_cutouts=args["bool_cutouts"], batchsize=args["batchsize"], ORDER_DATA=args["order_data"])

    try:
        # model
        if args["problem"] in ["1hp", "2stages", "test"]:
            model = UNet(in_channels=input_channels, out_channels=output_channels, depth=args["depth"], init_features=args["init_features"], kernel_size=args["kernel_size"]).float()
        elif args["problem"] in ["allin1"]:
            model = UNetNoPad2(in_channels=input_channels, out_channels=output_channels, depth=args["depth"], init_features=args["init_features"], kernel_size=args["kernel_size"], stride=args["stride"], dilation=args["dilation"], activation=args["activation_fct"], norm=args["norm"], repeat_inner=args["repeat_inner"]).float()
        model.to(args["device"])
        
        if args["case"] in ["test", "finetune"]:
            model.load(args["model"], args["device"])
        if args["case"] == "test":
            model.eval()

        if args["case"] in ["train", "finetune"]:
            loss_mapping = {
                "mae": L1Loss(),
                "mse": MSELoss()
            }
            loss = loss_mapping.get(args["train_loss"].lower(), MSELoss())
            solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=loss, finetune=(args["case"] == "finetune"), optimizer_switch=args["optimizer_switch"], learning_rate=float(args["lr"]))
            training_time = datetime.now()
            try:
                solver.load_lr_schedule(args["destination"] / "learning_rate_history.csv")
                val_loss = solver.train(args, optuna_trial=trial)
            except KeyboardInterrupt:
                logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
                val_loss = solver.best_model_params["loss"]
            finally:
                solver.save_lr_schedule(args["destination"] / "learning_rate_history.csv")
                print("Training finished")

            # save model and train metrics
            training_time = datetime.now() - training_time
            model.save(args["destination"] / "models", f"{run_name}.pt")
            # solver.save_metrics_to_overall_csv(args["destination"], model.num_of_params(), args["epochs"], training_time, args["device"])
            # metrics = solver.get_metrics_wandb(model, dataloaders, args, training_time.total_seconds(), vT_case)

        if args["case"] == "test":
            visualizations(model, dataloaders["val"], args, plot_path=args["destination"] / "val", amount_datapoints_to_visu=1, pic_format="png")
    except Exception as e:
        print(f"An error occurred: {e}")
        val_loss = 0.1

    # Clear up memory
    del model
    del dataloaders
    torch.cuda.empty_cache()

    return val_loss

def get_vT_case(output_channels):
    if output_channels == 1:
        vT_case = "temperature"
    elif output_channels == 2:
        vT_case = "velocities"
    return vT_case

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=str, default=None)
    parser.add_argument("--hsearch", type=bool, default=False)
    parser.add_argument("--problem", type=str, choices=["1hp", "2stages", "allin1", "extend", "test"], default="allin1")
    args = parser.parse_args()
    args = vars(args)

    hyperparameter_search = args["hsearch"]
    
    args["destination"] = Path(f"/scratch/sgs/pelzerja/runs/{args['problem']}") / args["destination"]
    current_destination = args["destination"] # just to make sure that nothing is overwritten
    args = read_cla(args["destination"])
    args["destination"] = current_destination
    
    # log_file = open(args["destination"]/"output.log", "w")
    # sys.stdout, sys.stderr = log_file, log_file

    if hyperparameter_search:
        print("Study name: ", args["destination"])
        study = optuna.create_study(direction="minimize", storage=f"sqlite:////scratch/sgs/pelzerja/runs/{args['problem']}/hps_2hpcnn.db", study_name="2025_09_09", load_if_exists=True)
        study.optimize(run, n_trials=50)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    else:
        print("Running with fixed hyperparameters")
        model = train_read_hyperparams(args)

    # log_file.close()
    
    print("Done")