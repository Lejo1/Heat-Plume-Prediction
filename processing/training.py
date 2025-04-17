import logging
import multiprocessing
import numpy as np
import torch
from torch.nn import MSELoss, L1Loss
from datetime import datetime

from preprocessing.data_init import init_data, load_all_datasets_in_full
from processing.networks.unet import UNet
from processing.networks.unetVariants import UNetHalfPad2, UNetNoPad2
from processing.solver import Solver
from postprocessing.visualization import visualizations
from postprocessing.measurements import measure_losses_paper24
from utils.utils_args import save_yaml
from processing.losses import CombiLoss
from neuralop.models import TFNO, FNO

def train(args: dict):
    np.random.seed(1)
    torch.manual_seed(1)
    multiprocessing.set_start_method("spawn", force=True)

    tmp_bool_cutouts = False
    input_channels, output_channels, dataloaders = init_data(args, tmp_bool_cutouts=tmp_bool_cutouts, batchsize=2)

    # model
    model = FNO(n_modes=(64,64), hidden_channels=40, n_layers=8, in_channels=input_channels, out_channels=output_channels)
    model.to(args["device"])
    
    if args["case"] in ["test", "finetune"]:
        model.load(args["model"], args["device"])
    if args["case"] == "test":
        model.eval()

    if args["case"] in ["train", "finetune"]:
        loss = MSELoss() 
        solver = Solver(model, dataloaders["train"], dataloaders["val"], loss_func=loss, finetune=(args["case"] == "finetune"), optimizer_switch=True)
        training_time = datetime.now()
        try:
            solver.load_lr_schedule(args["destination"] / "learning_rate_history.csv")
            solver.train(args)
        except KeyboardInterrupt:
            logging.warning(f"Manually stopping training early with best model found in epoch {solver.best_model_params['epoch']}.")
        finally:
            solver.save_lr_schedule(args["destination"] / "learning_rate_history.csv")
            print("Training finished")

        training_time = datetime.now() - training_time
        torch.save(model.state_dict(), args["destination"] / "fno_model.pt")

    if tmp_bool_cutouts:
        dataloaders = load_all_datasets_in_full(args)
    for case in ["train"]: #, "val", "test"]:
        visualizations(model, dataloaders[case], args, plot_path=args["destination"] / case, amount_datapoints_to_visu=1, pic_format="png")

    return model