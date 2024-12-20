import os
import pathlib
from dataclasses import dataclass
from typing import Dict
import yaml
from data_stuff.utils import SettingsTraining, get_splits
from data_stuff.dataset import SimulationDataset
import torch
from torch.utils.data import DataLoader, random_split


def init_data(settings: SettingsTraining, seed=1):
    dataset = SimulationDataset(settings.dataset_prep)
    print(f"Length of dataset: {len(dataset)}")
    generator = torch.Generator().manual_seed(seed)

    split_ratios = [0.7, 0.2, 0.1]
    if settings.case in ["test","visualize"]:
        split_ratios = [0.0, 0.0, 1.0]

    datasets = random_split(dataset, get_splits(len(dataset), split_ratios), generator=generator)
    dataloaders = {}
    try:
        dataloaders["train"] = DataLoader(datasets[0], batch_size=40, shuffle=True, num_workers=0)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=40, shuffle=True, num_workers=0)
    except: pass
    dataloaders["test"] = DataLoader(datasets[2], batch_size=10, shuffle=True, num_workers=0)

    return dataset.input_channels, dataloaders

def load_yaml(path: pathlib.Path, file_name="settings") -> Dict:
    with open(path / f"{file_name}.yaml", "r") as file:
        settings = yaml.safe_load(file)
    return settings


def save_yaml(settings: Dict, path: str, name_file: str = "settings"):
    path = pathlib.Path(path)
    with open(path / f"{name_file}.yaml", "w") as file:
        yaml.dump(settings, file)

@dataclass
class SettingsTraining:
    dataset_raw: str
    inputs: str
    device: str
    epochs: int
    destination: pathlib.Path = ""
    dataset_prep: str = ""
    case: str = "train"
    finetune: bool = False
    model: str = None
    test: bool = False
    case_2hp: bool = False
    prepare_xhp: bool = False
    visualize: bool = False
    only_prep: bool = False
    save_inference: bool = False
    architecture: str = "2stages"
    notes: str = ""
    skip_per_dir: int = 4
    len_box: int = 256
    
    def __post_init__(self):
        if self.case in ["finetune", "finetuning", "Finetune", "Finetuning"]:
            self.finetune = True
            self.case = "finetune"
            assert self.model is not None, "Path to model is not defined"
        elif self.case in ["test", "testing", "Test", "Testing", "TEST"]:
            self.case = "test"
            self.test = True
            assert self.finetune is False, "Finetune is not possible in test mode"
        elif self.case in ["train", "training", "Train", "Training", "TRAIN"]:
            self.case = "train"
            assert self.finetune is False, "Finetune is not possible in train mode"
            assert self.test is False, "Test is not possible in train mode"

        if self.case in ["test", "finetune"]:
            assert self.model != "runs/default", "Please specify model path for testing or finetuning"

        if self.destination == "":
            self.destination = self.dataset_raw + " inputs_" + self.inputs + " case_"+self.case + " box"+str(self.len_box) + " skip"+str(self.skip_per_dir)

    def save(self):
        save_yaml(self.__dict__, self.destination, "command_line_arguments")
        
    def make_destination_path(self, destination_dir: pathlib.Path):
        if self.destination == "":
            self.destination = self.dataset_raw + " inputs_" + self.inputs + " case_"+self.case + " box"+str(self.len_box) + " skip"+str(self.skip_per_dir)
        self.destination = destination_dir / self.destination
        self.destination.mkdir(exist_ok=True)

    def make_model_path(self, destination_dir: pathlib.Path):
        self.model = destination_dir / self.model

    def save_notes(self):
        # save notes to text file in destination
        if self.notes != "":
            with open(self.destination / "notes.txt", "w") as file:
                file.write(self.notes)