import csv
import logging
import pathlib
import time
from dataclasses import dataclass

from torch import manual_seed
from torch.nn import Module, modules, MSELoss, L1Loss, HuberLoss, SmoothL1Loss
from torch.optim import Adam, Optimizer, lr_scheduler, LBFGS
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from copy import deepcopy

from postprocessing.visualization import visualizations
from processing.networks.unetVariants import UNetNoPad2
from processing.networks.model import weights_init
from utils.utils_args import save_yaml
from processing.losses import CombiLoss, SSIMLoss, KLD_log, LinfLoss, IoULoss
from torchmetrics.regression import MeanAbsolutePercentageError as MAPE

@dataclass
class Solver(object):
    model: Module
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    loss_func: modules.loss._Loss = MSELoss()
    learning_rate: float = 1e-4
    opt: Optimizer = Adam
    finetune: bool = False
    best_model_params: dict = None
    metrics: dict = None

    def __post_init__(self):
        self.opt = self.opt(self.model.parameters(),self.learning_rate, weight_decay=1e-4)
        # contains the epoch and learning rate, when lr changes
        self.lr_schedule = {0: self.opt.param_groups[0]["lr"]}
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.opt, patience=5, cooldown=10, factor=0.5)

        if not self.finetune:
            self.model.apply(weights_init)
        
        self.metrics: dict = {"MSE": MSELoss(), "MAE": L1Loss(), "Linf": LinfLoss(), "Huber": HuberLoss(), "KLD": KLD_log(), "SmoothL1": SmoothL1Loss(), "SSIM": SSIMLoss(), "MAPE": MAPE(), "IoU": IoULoss()} #, "X-MSE": None, "Y-MSE": None}

    def train(self, args: dict):
        manual_seed(0)
        start_time = time.perf_counter()
        # initialize tensorboard
        writer = SummaryWriter(args["destination"])
        device = args["device"]
        # writer.add_graph(self.model, next(iter(self.train_dataloader))[0].to(device))

        epochs = tqdm(range(args["epochs"]), desc="epochs", disable=False)
        for epoch in epochs:
            try:
                # Set lr according to schedule
                if epoch in self.lr_schedule.keys():
                    self.opt.param_groups[0]["lr"] = self.lr_schedule[epoch]
                    
                # Validation
                # if epoch % 10 == 0:
                self.model.eval()
                val_epoch_loss, other_losses_val = self.run_epoch(self.val_dataloader, device)

                # Training
                self.model.train()
                train_epoch_loss, other_losses_train = self.run_epoch(self.train_dataloader, device)

                # if epoch % 10 == 0:
                for metric_name, metric_value in other_losses_val.items():
                    writer.add_scalar(f"val {metric_name}", metric_value, epoch)
                for metric_name, metric_value in other_losses_train.items():
                        writer.add_scalar(f"train {metric_name}", metric_value, epoch)

                # Logging
                writer.add_scalar("train_loss", train_epoch_loss, epoch)
                writer.add_scalar("val_loss", val_epoch_loss, epoch)
                writer.add_scalar("learning_rate", self.opt.param_groups[0]["lr"], epoch)
                epochs.set_postfix_str(f"train loss: {train_epoch_loss:.2e}, val loss: {val_epoch_loss:.2e}, lr: {self.opt.param_groups[0]['lr']:.1e}")
                
                # Keep best model
                if self.best_model_params is None or val_epoch_loss < self.best_model_params["loss"]:
                    self.best_model_params = {
                        "epoch": epoch,
                        "loss": val_epoch_loss,
                        "train loss": train_epoch_loss,
                        # "val RMSE": val_epoch_loss**0.5, # TODO only true if loss_func == MSELoss()
                        # "train RMSE": train_epoch_loss**0.5, #TODO only true if loss_func == MSELoss()
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.opt.state_dict(),
                        "parameters": self.model.parameters(),
                        "training time in sec": (time.perf_counter() - start_time),
                    }
                    if False:
                        self.model.save(args["destination"], model_name=f"best_model_e{epoch}.pt")

                # self.lr_scheduler.step(val_epoch_loss)

            except KeyboardInterrupt:
                model_tmp = UNetNoPad2(in_channels=3, out_channels=2, depth=4, init_features=32, kernel_size=5)
                model_tmp.load_state_dict(self.best_model_params["state_dict"])
                model_tmp.to(args["device"])
                model_tmp.save(args["destination"], model_name=f"interim_model_e{epoch}.pt")
                visualizations(model_tmp, self.val_dataloader, args, plot_path=args["destination"] / f"plot_val_interim_e{epoch}", amount_datapoints_to_visu=2, pic_format="png")

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

    def run_epoch(self, dataloader: DataLoader, device: str):
        epoch_loss = 0.0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            if self.model.training:
                self.opt.zero_grad()

            y_pred = self.model(x)
            required_size = y_pred.shape[2:]
            start_pos = ((y.shape[2] - required_size[0])//2, (y.shape[3] - required_size[1])//2)
            y_reduced = y[:, :, start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]]

            loss = self.loss_func(y_pred, y_reduced)

            if self.model.training:
                loss.backward()
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
            
        return epoch_loss, metric_values

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

    def save_metrics_separate_yaml(self, destination: pathlib.Path, no_params:int, max_epochs:int, training_time:float, device: str = "cpu"):
        # prepare data as dict

        self.model.eval()
        train_epoch_loss, other_losses_train = self.run_epoch(self.train_dataloader, device)
        val_epoch_loss, other_losses_val = self.run_epoch(self.val_dataloader, device)

        metrics = {}
        metrics["no_params"] = no_params
        metrics["max_epochs"] = max_epochs
        metrics["training_time [s]"] = training_time
        metrics["best_epoch"] = self.best_model_params["epoch"]
        metrics["train"] = other_losses_train
        metrics["train"]["train loss"] = train_epoch_loss
        metrics["val"] = other_losses_val
        metrics["val"]["val loss"] = val_epoch_loss

        # save data as yaml
        save_yaml(metrics, destination / "measurements.yaml")


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

    def get_metrics_wandb(self, model: Module, dataloaders: dict, args: dict, training_time: float, vT_case: str = "vT"):
        """ get metrics for wandb """
        self.model.eval()
        metrics = {}
        for case, dataloader in dataloaders.items():
            if case in ["train", "test"]:
                continue
            for x, y in dataloader:
                x = x.to(args["device"])
                y = y.to(args["device"])

                y_pred = model(x)
                required_size = y_pred.shape[2:]
                start_pos = ((y.shape[2] - required_size[0])//2, (y.shape[3] - required_size[1])//2)
                y_reduced = y[:, :, start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]]
                # Calculate metrics
                if vT_case == "velocities":
                    metrics[f"{case} X-MSE"] = MSELoss()(y_pred[:, 0, :, :], y_reduced[:, 0, :, :]).detach().item()
                    metrics[f"{case} Y-MSE"] = MSELoss()(y_pred[:, 1, :, :], y_reduced[:, 1, :, :]).detach().item()
                for name, metric in self.metrics.items():
                    if name == "X-MSE" or name == "Y-MSE":
                        continue
                    try:
                        metrics[f"{case} {name}"] = metric(y_pred, y_reduced).detach().item()
                    except:
                        metrics[f"{case} {name}"] = metric(y_pred.cpu(), y_reduced.cpu())
                metrics[f"{case} RMSE"] = MSELoss()(y_pred, y_reduced).detach().item()**0.5
            metrics["No. params"] = self.model.num_of_params()
            metrics["Best epoch"] = self.best_model_params["epoch"]
            metrics["Training time in s"] = training_time

        return metrics