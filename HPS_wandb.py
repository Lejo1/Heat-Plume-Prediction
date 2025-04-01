import wandb
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
from processing.solver import Solver
from postprocessing.visualization import visualizations
from postprocessing.measurements import measure_losses_paper24
from utils.utils_args import save_yaml
import utils.utils_args as ut
import preprocessing.preprocessing as prep
from main import read_cla, train

def run():
    with wandb.init(): #, tags=[model_name]):
        config = wandb.config
        run_name = wandb.run.name
        np.random.seed(1)
        torch.manual_seed(1)
        multiprocessing.set_start_method("spawn", force=True)

        args["data_prep"] = f"/scratch/sgs/pelzerja/datasets_prepared/allin1/dataset_100hp_giant_real_fixP0_0025 inputs_{config.inputs} outputs_xy"

        ut.make_paths(args) # and check if data / model exists
        ut.save_yaml(args, args["destination"] / "command_line_arguments.yaml")

        # prepare data
        prep.preprocessing(args) # and save info.yaml in model folder

        args["inputs"] = config.inputs
        args["len_box"] = config.len_box
        args["skip_per_dir"] = config.skip_per_dir
        args["stride"] = config.stride
        args["dilation"] = config.dilation
        args["activation_fct"] = config.activation_fct
        args["norm"] = config.norm
        args["repeat_inner"] = config.repeat_inner
        args["optimizer_switch"] = config.optimizer_switch

        tmp_bool_cutouts = config.bool_cutouts
        input_channels, output_channels, dataloaders = init_data(args, tmp_bool_cutouts=tmp_bool_cutouts, batchsize=config.batchsize)
        vT_case = get_vT_case(output_channels)

        # model
        model = UNetNoPad2(in_channels=input_channels, out_channels=output_channels, depth=config.depth, init_features=config.init_features, kernel_size=config.kernel_size, stride=args["stride"], dilation=args["dilation"], activation=args["activation_fct"], norm=args["norm"], repeat_inner=args["repeat_inner"]).float()
        model.to(args["device"])
        
        if args["case"] in ["test", "finetune"]:
            model.load(args["model"], args["device"])
        if args["case"] == "test":
            model.eval()

        if args["case"] in ["train", "finetune"]:
            loss = L1Loss() if config.train_loss.lower() == "mae" else MSELoss()
            solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=loss, finetune=(args["case"] == "finetune"), optimizer_switch=args["optimizer_switch"])
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
            model.save(args["destination"] / "models", f"{run_name}.pt")
            solver.save_metrics_to_overall_csv(args["destination"], model.num_of_params(), args["epochs"], training_time, args["device"])
            metrics = solver.get_metrics_wandb(model, dataloaders, args, training_time.total_seconds(), vT_case)

            wandb.log(metrics)
    wandb.finish()

def get_vT_case(output_channels):
    if output_channels == 1:
        vT_case = "temperature"
    elif output_channels == 2:
        vT_case = "velocities"
    return vT_case

def set_device_environment(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args["device"] if not args["device"]=="cpu" else "" #e.g. "1"
    args["device"] = torch.device(f"cuda:{args['device']}" if not args["device"]=="cpu" else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, choices=["1hp", "2stages", "allin1", "extend", "test"], default="allin1")
    parser.add_argument("--data_raw", type=str, default="dataset_small_10dp_varyK", help="Name of the raw dataset (without inputs)")
    parser.add_argument("--data_prep", type=str, default=None)
    parser.add_argument("--allin1_prepro_n_case", type=str, choices=["gt", "unet", "cdmlp"], default=None, help="Case for preprocessing of allin1")
    parser.add_argument("--inputs", type=str, default="gksi") #e.g. "gki", "gksi100", "ogksi1000_finetune", "t", "lmi", "lmik","lmikp", ...
    parser.add_argument("--outputs", type=str, default="t") # e.g. "t" for allin1 step2; "xy" for allin1 step1; "xyt" for preparation
    parser.add_argument("--len_box", type=int, default=64) # for 1hp:256, extend:128?
    parser.add_argument("--skip_per_dir", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--case", type=str, choices=["train", "test", "finetune"], default="train")
    parser.add_argument("--model", type=str, default=None) # required for testing or finetuning
    parser.add_argument("--destination", type=str, default=None)
    parser.add_argument("--visualize", type=bool, default=False)
    parser.add_argument("--device", type=str, default="3", help="cuda device number or 'cpu'")
    parser.add_argument("--notes", type=str, default=None)
    args = parser.parse_args()
    args = vars(args)

    set_device_environment(args)

    hyperparameter_search = True
    #always read from file
    args["destination"] = Path(f"/scratch/sgs/pelzerja/runs/{args['problem']}") / args["destination"]
    current_destination = args["destination"]
    args = read_cla(args["destination"])
    args["destination"] = current_destination # just to make sure that nothing is overwritten
    ""
    if hyperparameter_search:
        os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "1"
        sweep_config = {
            "method": "bayes",
            "name": "extended",
            "metric": {"name": "val MSE", "goal": "minimize"},
            }

        sweep_config['parameters'] = ut.load_yaml(args["destination"] / "HPS_options.yaml")
        sweep_id = wandb.sweep(sweep_config, entity='jupelzer-university-of-stuttgart', project="hps_realK")
        args_global = args
        wandb.agent(sweep_id, function=run)
    else:
        model = train(args)

    print("Done")
