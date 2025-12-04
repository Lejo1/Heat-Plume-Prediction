import csv
import logging
import pathlib
import time
from dataclasses import dataclass
import wandb
import optuna

from math import isnan
from torch import manual_seed
from torch.nn import Module, modules, MSELoss, L1Loss, HuberLoss, SmoothL1Loss
from torch.optim import Adam, Optimizer, lr_scheduler, LBFGS
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.regression import MeanAbsolutePercentageError as MAPE
from tqdm.auto import tqdm
from copy import deepcopy

from utils.utils_args import save_yaml
from processing.networks.unetVariants import UNetNoPad2
from processing.networks.model import weights_init
from processing.losses import CombiLoss, SSIMLoss, KLD_log, LinfLoss, IoULoss
from postprocessing.visualization import visualizations

@dataclass
class Solver(object):
    model: Module
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    loss_func: modules.loss._Loss = MSELoss()
    learning_rate: float = 1e-4
    opt: Optimizer = Adam
    optimizer_switch: bool = False
    finetune: bool = False
    best_model_params: dict = None
    metrics: dict = None

    def __post_init__(self):
        self.opt = self.opt(self.model.parameters(),self.learning_rate, weight_decay=1e-4)
        # contains the epoch and learning rate, when lr changes
        self.lr_schedule = {0: self.opt.param_groups[0]["lr"]}
        # self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.opt, patience=5, cooldown=10, factor=0.5)

        if not self.finetune:
            self.model.apply(weights_init)
        
        self.metrics: dict = {"Huber": HuberLoss(), "MSE": MSELoss(), "MAE": L1Loss()} #, "Linf": LinfLoss(), "KLD": KLD_log(), "SmoothL1": SmoothL1Loss(), "SSIM": SSIMLoss(), "MAPE": MAPE(), "IoU": IoULoss()} #, "X-MSE": None, "Y-MSE": None}

    def train(self, args: dict, optuna_trial=None):
        # manual_seed(0)
        start_time = time.perf_counter()
        # initialize tensorboard
        if optuna_trial:
            log_dir = args["destination"] / f"trial{optuna_trial.number}"
            pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        else:
            log_dir = args["destination"]
        writer = SummaryWriter(log_dir)
        device = args["device"]

        # if optimizer_switch is True, switch to LBFGS optimizer after 90% of epochs
        self.epoch_switch_optimizer = args["epochs"] + 1
        if self.optimizer_switch:
            self.epoch_switch_optimizer = int(0.9 * self.epoch_switch_optimizer)

        epochs = tqdm(range(args["epochs"]), desc="epochs", disable=False)
        for epoch in epochs:
            try:
                if epoch == self.epoch_switch_optimizer:
                    # switch to LBFGS optimizer
                    self.opt = LBFGS(self.model.parameters(), history_size=20, line_search_fn="strong_wolfe")
                    logging.info(f"Switched to LBFGS optimizer at epoch {epoch}.")

                # Set lr according to schedule
                if epoch in self.lr_schedule.keys():
                    self.opt.param_groups[0]["lr"] = self.lr_schedule[epoch]
                    
                # # re-construct dataloaders
                # self.train_dataloader, self.val_dataloader = const_dataloaders(args, {"train": self.train_dataloader, "val": self.val_dataloader}).values()
                
                # Training
                self.model.train()
                train_epoch_loss, _ = self.run_epoch(self.train_dataloader, device) #, other_losses_train

                # Validation
                self.model.eval()
                _, other_losses_val = self.run_epoch(self.val_dataloader, device)
                # val_epoch_loss = other_losses_val["Huber"] # TODO change back? currently for realK
                val_epoch_loss = other_losses_val["MAE"] # TODO change back? currently for dummyK
                # val_epoch_loss, _ = self.run_epoch(self.val_dataloader, device) #TODO used for randomK

                # Logging
                writer.add_scalar("train_loss", train_epoch_loss, epoch)
                writer.add_scalar("val_loss", val_epoch_loss, epoch)
                # for metric_name, metric_value in other_losses_train.items():
                #     writer.add_scalar(f"train_{metric_name}", metric_value, epoch)
                # for metric_name, metric_value in other_losses_val.items():
                #     writer.add_scalar(f"val_{metric_name}", metric_value, epoch)
                writer.add_scalar("learning_rate", self.opt.param_groups[0]["lr"], epoch)
                epochs.set_postfix_str(f"train loss: {train_epoch_loss:.2e}, val loss: {val_epoch_loss:.2e}, lr: {self.opt.param_groups[0]['lr']:.1e}")
                
                # Keep best model
                if self.best_model_params is None or val_epoch_loss < self.best_model_params["loss"]:
                    self.best_model_params = {
                        "epoch": epoch,
                        "loss": val_epoch_loss,
                        "train loss": train_epoch_loss,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.opt.state_dict(),
                        "parameters": self.model.parameters(),
                        "training time in sec": (time.perf_counter() - start_time),
                    }
                    if False:
                        self.model.save(log_dir, model_name=f"best_model_e{epoch}.pt")

                # self.lr_scheduler.step(val_epoch_loss)

                if optuna_trial:
                    optuna_trial.report(val_epoch_loss, epoch)

                    # # Handle pruning based on the intermediate value.
                    # if optuna_trial.should_prune():
                    #     raise optuna.exceptions.TrialPruned()

            except KeyboardInterrupt:
                try:
                    model_tmp = deepcopy(self.model)
                    model_tmp.load_state_dict(self.best_model_params["state_dict"])
                    model_tmp.to(args["device"])
                    model_tmp.save(log_dir, model_name=f"interim_model_e{epoch}.pt")
                    visualizations(model_tmp, self.val_dataloader, args, plot_path=log_dir / f"plot_val_interim_e{epoch}", amount_datapoints_to_visu=2, pic_format="png")
                except Exception as e:
                    logging.error(e)
                    
                try:
                    new_lr = float(input("\nNew learning rate: "))
                except ValueError as e:
                    print(e)
                else:
                    for g in self.opt.param_groups:
                        g["lr"] = new_lr
                    self.lr_schedule[epoch] = self.opt.param_groups[0]["lr"]

        # Apply best model params to model
        self.model.load_state_dict(self.best_model_params["state_dict"]) #self.model = 
        self.opt.load_state_dict(self.best_model_params["optimizer"]) #self.opt =
        print(f"Best model was found in epoch {self.best_model_params['epoch']}.")

        return self.best_model_params["loss"]

    def run_epoch(self, dataloader: DataLoader, device: str):
        epoch_loss = 0.0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            if self.model.training:
                self.opt.zero_grad()
                if self.opt.__class__.__name__ == "LBFGS":
                    def closure():
                        """ closure function for optimizer LBFGS """
                        self.opt.zero_grad()
                        y_pred = self.model(x)
                        required_size = y_pred.shape[2:]
                        start_pos = ((y.shape[2] - required_size[0])//2, (y.shape[3] - required_size[1])//2)
                        y_reduced = y[:, :, start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]]

                        loss = self.loss_func(y_pred, y_reduced)
                        loss.backward()
                        return loss
                    self.opt.step(closure)

            y_pred = self.model(x)
            required_size = y_pred.shape[2:]
            start_pos = ((y.shape[2] - required_size[0])//2, (y.shape[3] - required_size[1])//2)
            y_reduced = y[:, :, start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]]

            loss = self.loss_func(y_pred, y_reduced)

            if self.model.training:
                loss.backward()
                if self.opt.__class__.__name__ == "LBFGS":
                    self.opt.step(closure)
                else:
                    self.opt.step()

            epoch_loss += loss.detach().item()
        epoch_loss /= len(dataloader)

        # Calculate metrics
        metric_values = {}
        for metric_name, metric in self.metrics.items():
            if metric_name == "X-MSE":
                metric_values[metric_name] = MSELoss()(y_pred[:, 0, :, :], y_reduced[:, 0, :, :]).detach().item()
            elif metric_name == "Y-MSE":
                metric_values[metric_name] = MSELoss()(y_pred[:, 1, :, :], y_reduced[:, 1, :, :]).detach().item()
            else:
                try:
                    metric_values[metric_name] = metric(y_pred, y_reduced).detach().item()
                except:
                    metric_values[metric_name] = metric(y_pred.cpu(), y_reduced.cpu())
                    # necessary exception for MAPE (some error in the function requires to put all to cpu) # necessary exception for SSIM, because it requires to detach before calculating this loss
            
        return epoch_loss, metric_values #None #, 

    def save_lr_schedule(self, path: str):
        """ save learning rate history to csv file"""
        with open(path, "w") as f:
            logging.info(f"Saving lr-schedule to {path}.")
            for epoch, lr in self.lr_schedule.items():
                f.write(f"{epoch},{lr}\n")

    def load_lr_schedule(self, path: pathlib.Path, case_2hp:bool=False):
        """ read lr-schedule from csv file"""
        # check if path contains lr-schedule, else use default one
        if not path.exists():
            logging.warning(f"Could not find lr-schedule at {path}. Using default lr-schedule instead.")
            path = pathlib.Path.cwd() / "processing" / "lr_schedules"
            lr_schedule_file = "default_lr_schedule.csv" if not case_2hp else "default_lr_schedule_2hp.csv"
            path = path / lr_schedule_file

        with open(path, "r") as f:
            for line in f:
                epoch, lr = line.split(",")
                self.lr_schedule[int(epoch)] = float(lr)

    def save_metrics_separate_yaml(self, args: dict, training_time:float):
        device = args["device"]

        self.model.eval()
        train_epoch_loss, other_losses_train = self.run_epoch(self.train_dataloader, device) #other_losses_train
        val_epoch_loss, other_losses_val = self.run_epoch(self.val_dataloader, device) #other_losses_val

        # prepare data as dict
        metrics = {}
        metrics["no_params"] = self.model.num_of_params()
        metrics["max_epochs"] = args["epochs"]
        metrics["training_time [s]"] = training_time
        metrics["best_epoch"] = self.best_model_params["epoch"]
        metrics["train"] = other_losses_train
        metrics["train loss"] = train_epoch_loss #["train"]
        metrics["val"] = other_losses_val
        metrics["val loss"] = val_epoch_loss #["val"]

        # save data as yaml
        save_yaml(metrics, args["destination"] / "measurements.yaml")


    def save_metrics_to_overall_csv(self, destination: pathlib.Path, no_params:int, max_epochs:int, training_time:float, device: str = "cpu"):
        csv_file = open(destination.parent / "measurements_all_metrics.csv", "a")

        # no. parameters, max. epochs, training time
        csv_writer = csv.writer(csv_file)

        self.model.eval()
        train_epoch_loss, other_losses_train = self.run_epoch(self.train_dataloader, device)
        val_epoch_loss, other_losses_val = self.run_epoch(self.val_dataloader, device)

        other_losses_train_list = []
        for _, val in other_losses_train.items():
            other_losses_train_list.append(val)
        other_losses_val_list = []
        for _, val in other_losses_val.items():
            other_losses_val_list.append(val)

        row = [destination.name, self.best_model_params["epoch"], train_epoch_loss, val_epoch_loss]
        row.extend(other_losses_train_list)
        row.extend(other_losses_val_list)
        row.extend([no_params, max_epochs, training_time])
               
        csv_writer.writerow(row)
        csv_file.close()