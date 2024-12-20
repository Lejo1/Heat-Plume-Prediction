from functools import partial
from networks.unet import UNet, UNetBC
from networks.unetParallel import UNetParallel
from networks.unetQuad import UNetQuad
import torch
from torch.nn import Module, MSELoss, modules
from torch.optim import Adam, Optimizer, RMSprop
from pathlib import Path
from ray.train import Checkpoint, get_checkpoint
import ray.cloudpickle as pickle
import tempfile
from ray import tune
from ray import train
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import random_split
import os
from torch.utils.tensorboard import SummaryWriter
from torch import manual_seed
from tqdm.auto import tqdm
from data_stuff.utils import SettingsTraining, init_data
import csv
from torch.utils.data import DataLoader
from ray.train import RunConfig
from ray.tune.search.optuna import OptunaSearch
from data_stuff.dataset import SimulationDataset, DatasetExtend1, DatasetExtend2, get_splits

def tune_nn(settings: SettingsTraining, num_samples=200, max_num_epochs=20, ):
    """
    method to start hyperparameter tuning
    
    settings file contains the architecture information and training data info
    num_samples: how many hyperparameter configurations should be sampled
    max_num_epochs: how many epochs should the sampled configs be trained for
    """
    if settings.architecture == "parallel":
        kernel_sizes = [(4*i,i) for i in range(2,5)]
        kernel_sizes = kernel_sizes + [(2*j,j) for j in range(2,5)]
        config = {
            "features": tune.choice([32,64]),
            "lr": tune.choice([1e-4]),
            "depth": tune.choice([5]),
            "kernel_size": tune.choice([5]),
            "weight_decay": tune.choice([1e-5]),
            "padding_mode": tune.choice(['replicate']),
            "par_depth": tune.choice([1,2,3]),
            "par_dil": tune.choice([(6,1),(4,1),(2,1),(1,1),(6,2),(4,2),(2,2)]),
            "par_kern": tune.choice([(4,1),(8,2),(12,3),(16,4),(2,1),(4,2),(8,4),(16,8),(1,4),(2,8),(3,12)]),
        }
    elif settings.architecture == "quad":
        config = {
            "features": tune.choice([2**i for i in range(4,7)]),
            "lr": tune.choice([1e-4]),
            "depth": tune.choice([2,3,4,5]),
            "kernel_size": tune.choice([3,4,5,6,7]),
            "weight_decay": tune.choice([1e-5]),
            "padding_mode": tune.choice(['replicate']),
            "dilation": tune.choice([1]),
            "down_kernel": tune.choice([1])
        }
    else:
        config = {
            "features": tune.choice([2**i for i in range(4,7)]),
            "lr": tune.choice([1e-4]),
            "depth": tune.choice([2,3,4,5]),
            "kernel_size": tune.choice([3,4,5,6,7]),
            "weight_decay": tune.choice([1e-5]),
            "padding_mode": tune.choice(['zeros','replicate']),
            "dilation": tune.choice([1,2,3])
        }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    algo = OptunaSearch()
    trainable_with_resources = tune.with_resources(partial(train_config, settings=settings), {"cpu": 4, "gpu": 1})
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            search_alg=algo,
            num_samples=num_samples,
            max_concurrent_trials=1,
        ),
        run_config=train.RunConfig(
            stop={"training_iteration": max_num_epochs},
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best config is:", results.get_best_result().config)
    print("path is:", results.get_best_result().path)
    print("metrics are", results.get_best_result().metrics_dataframe)
    # result = tune.run(
    #     partial(train_config, settings=settings),
    #     resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
    #     config=config,
    #     num_samples=num_samples,
    #     scheduler=scheduler,
    #     search_alg=algo,
    # )

    #best_trial = results.get_best_trial("loss", "min", "last")
    #print(f"Best trial config: {best_trial.config}")
    #print(f"Best trial final validation loss: {best_trial.last_result['loss']}")


def train_config(config,settings=None):
    """
    trains a model based on the configurations defined by config
    """
    torch.cuda.empty_cache()
    input_channels, dataloaders = init_data(settings)
    if settings.architecture == "parallel":
        model = UNetParallel(in_channels=input_channels,
                            init_features=config["features"],
                            depth=config["depth"],
                            padding_mode=config["padding_mode"],
                            dilation=1,
                            par_depth=config["par_depth"],
                            par_dil=config["par_dil"],
                            par_kern=config["par_kern"]).to(settings.device)
    elif settings.architecture == "quad":
        model = UNetQuad(in_channels=input_channels,init_features=config["features"],depth=config["depth"],padding_mode=config["padding_mode"],dilation=config["dilation"],down_kernel=config["down_kernel"]).to(settings.device)
    else:
        model = UNet(in_channels=input_channels,init_features=config["features"],depth=config["depth"],padding_mode=config["padding_mode"],dilation=config["dilation"]).to(settings.device)
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_function = MSELoss()
    device = settings.device
    model = model.to(device)

    train_dataloader = dataloaders["train"]

    epochs = tqdm(range(settings.epochs), desc="epochs", disable=False)
    while True:

        # Training
        model.train()
        train_loss = 0.0
        for x, y,fname in train_dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_pred = model(x)

            loss = None
            loss =  loss_function(y_pred, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()
        train_loss /= len(train_dataloader)
    
        loss = test_loss(model, dataloaders["test"], settings.device)
        train.report({"loss": loss})


        


def test_loss(model: UNet, dataloader: DataLoader, device: str, loss_func: modules.loss._Loss = MSELoss()):
    model.eval()
    mse_loss = 0.0

    for x, y, fname in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x).to(device)
        mse_loss += loss_func(y_pred, y).detach().item()
        
    mse_loss /= len(dataloader)

    return mse_loss



