import logging
from pathlib import Path
import time
from dataclasses import dataclass
import torch
from torch import manual_seed
from torch.nn import Module, modules, MSELoss, HuberLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, Optimizer, LBFGS
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from copy import deepcopy

from postprocessing.visualization import visualizations
from processing.networks.model import weights_init
from utils.utils_args import save_yaml

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
    clip_grad_norm: float = None  # cap on the global gradient L2 norm, e.g. 1.0 for end-to-end training (exploding gradients through the streamlines); None = off
    pipeline_tap: object = None  # optional PipelineTap: every N steps, plot each stage's in/out + the loss gradients around it
    epoch_callback: object = None  # optional fn(epoch) run at the start of each epoch, e.g. the v_blur annealing schedule

    def __post_init__(self):
        self.opt = self.opt(self.model.parameters(),self.learning_rate, weight_decay=1e-4)
        # contains the epoch and learning rate, when lr changes
        self.lr_schedule = {0: self.opt.param_groups[0]["lr"]}

        if not self.finetune:
            self.model.apply(weights_init)
        
        self.metrics: dict = {"Huber": HuberLoss(), }

    def train(self, args: dict):
        manual_seed(0)
        start_time = time.perf_counter()
        # initialize tensorboard
        writer = SummaryWriter(args["destination"])
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

                # schedules that are not the lr (v_blur annealing); runs before train and val, so
                # both halves of the epoch see the same forward model
                if self.epoch_callback is not None:
                    self.epoch_callback(epoch)
                    
                # Training
                self.model.train()
                train_epoch_loss, other_losses_train = self.run_epoch(self.train_dataloader, device) #, 
                
                # Validation
                self.model.eval()
                val_epoch_loss, other_losses_val = self.run_epoch(self.val_dataloader, device)
                if False: # realK
                    val_epoch_loss = other_losses_val["Huber"] # TODO for realK


                # # Logging
                for metric_name, metric_value in other_losses_val.items():
                    writer.add_scalar(f"val {metric_name}", metric_value, epoch)
                for metric_name, metric_value in other_losses_train.items():
                        writer.add_scalar(f"train {metric_name}", metric_value, epoch)

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
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.opt.state_dict(),
                        "parameters": self.model.parameters(),
                        "training time in sec": (time.perf_counter() - start_time),
                    }

            except KeyboardInterrupt:
                # allows to interrupt training with ctrl+c to change the lr manually
                try:
                    model_tmp = deepcopy(self.model)
                    model_tmp.load_state_dict(self.best_model_params["state_dict"])
                    model_tmp.to(args["device"])
                    model_tmp.save(args["destination"], model_name=f"interim_model_e{epoch}.pt")
                    visualizations(model_tmp, self.val_dataloader, args, plot_path=args["destination"] / f"plot_val_interim_e{epoch}", amount_datapoints_to_visu=2, pic_format="png")
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
        self.model.load_state_dict(self.best_model_params["state_dict"]) 
        self.opt.load_state_dict(self.best_model_params["optimizer"])
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

            # arm the pipeline diagnostic BEFORE the forward pass (it has to retain the stage
            # tensors while the graph is built); it only observes, the step itself is unchanged
            armed = self.pipeline_tap.arm(self.model, x) if (self.pipeline_tap is not None and self.model.training) else False

            y_pred = self.model(x)
            required_size = y_pred.shape[2:]
            start_pos = ((y.shape[2] - required_size[0])//2, (y.shape[3] - required_size[1])//2)
            y_reduced = y[:, :, start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]]

            loss = self.loss_func(y_pred, y_reduced)

            if self.model.training:
                loss.backward()
                # plot while .grad still holds the gradients as produced (clipping rescales them)
                if armed:
                    self.pipeline_tap.plot(self.model, y_reduced, loss)
                if self.pipeline_tap is not None:
                    self.pipeline_tap.advance()
                if self.clip_grad_norm is not None:
                    # zero out non-finite entries first: clip_grad_norm_ with an inf total norm
                    # would scale by 0 and turn inf entries into NaN (inf * 0), poisoning the weights
                    for p in self.model.parameters():
                        if p.grad is not None:
                            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                if self.opt.__class__.__name__ == "LBFGS":
                    self.opt.step(closure)
                else:
                    self.opt.step()

            epoch_loss += loss.detach().item()
        epoch_loss /= len(dataloader)

        # Calculate metrics
        metric_values = {}
        for metric_name, metric in self.metrics.items():
            metric_values[metric_name] = metric(y_pred, y_reduced).detach().item()

        return epoch_loss, metric_values

    def save_lr_schedule(self, path: str):
        """ save learning rate history to csv file"""
        with open(path, "w") as f:
            logging.info(f"Saving lr-schedule to {path}.")
            for epoch, lr in self.lr_schedule.items():
                f.write(f"{epoch},{lr}\n")

    def load_lr_schedule(self, path: Path):
        """ read lr-schedule from csv file"""
        # check if path contains lr-schedule, else use default one
        if not path.exists():
            logging.warning(f"Could not find lr-schedule at {path}. Using default lr-schedule instead.")
            path = Path.cwd() / "processing" / "default_lr_schedule.csv"

        with open(path, "r") as f:
            for line in f:
                epoch, lr = line.split(",")
                self.lr_schedule[int(epoch)] = float(lr)

    def save_metrics_separate_yaml(self, destination: Path, no_params:int, max_epochs:int, training_time:float, device: str = "cpu"):
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