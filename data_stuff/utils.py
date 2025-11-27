import os
import pathlib
from dataclasses import dataclass
from typing import Dict
import yaml


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
    prev_boxes: int
    extend: int
    overfit_str: str
    destination: pathlib.Path = ""
    dataset_prep: str = ""
    case: str = "train"
    finetune: bool = False
    model: str = None
    test: bool = False
    case_2hp: bool = False
    visualize: bool = False
    save_inference: bool = False
    problem: str = "2stages"
    notes: str = ""
    skip_per_dir: int = 4
    len_box: int = 640
    net: str = "convLSTM"
    vis_entire_plume: bool = False
    overfit: int = 0
    num_layers: int = 1
    loss: str = "mse"
    activation: str = "relu"
    enc_conv_features = [16, 32, 64, 64, 64]
    dec_conv_features = [64, 64, 64]
    enc_kernel_sizes = [7, 5, 5, 5, 5]
    dec_kernel_sizes = [5, 5, 7]

    def __post_init__(self):
        # Normalize the case field and set associated flags.
        case_map = {
            "finetune": ["finetune", "finetuning", "Finetune", "Finetuning"],
            "test": ["test", "testing", "Test", "Testing", "TEST"],
            "train": ["train", "training", "Train", "Training", "TRAIN"],
        }
        
        if self.case in case_map["finetune"]:
            self.finetune = True
            self.case = "finetune"
            assert self.model is not None, "Path to model is not defined for finetuning"
        elif self.case in case_map["test"]:
            self.case = "test"
            self.test = True
            assert not self.finetune, "Finetune is not possible in test mode"
        elif self.case in case_map["train"]:
            self.case = "train"
            assert not self.finetune, "Finetune is not possible in train mode"
            assert not self.test, "Test is not possible in train mode"
        else:
            raise ValueError(f"Invalid case: {self.case}")

        # Additional validation for test and finetune cases.
        if self.case in ["test", "finetune"]:
            assert self.model != "runs/default", "Please specify a valid model path for testing or finetuning"

        # Initialize overfit_str based on the overfit parameter.
        self.overfit_str = f" overfit_{self.overfit}" if self.overfit else ""

        # Set the default destination if none is provided.
        if not self.destination:
            self.destination = pathlib.Path(
                f"case_{self.case} prev_{self.prev_boxes} extend_{self.extend} "
                f"skip_{self.skip_per_dir} loss_{self.loss} layers_{self.num_layers}"
            )       

    def save(self):
        save_yaml(self.__dict__, self.destination, "command_line_arguments")
        
    def make_destination_path(self, destination_dir: pathlib.Path):
        if self.destination == "":
            self.destination = f"case_{self.case} prev_{self.prev_boxes} extend_{self.extend} skip_{self.skip_per_dir} loss_{self.loss} layers_{self.num_layers}"
        self.destination = destination_dir / self.destination
        self.destination.mkdir(parents=True, exist_ok=True)

    def make_model_path(self, destination_dir: pathlib.Path):
        self.model = destination_dir / self.model

    def save_notes(self):
        # save notes to text file in destination
        if self.notes is not None:
            with open(self.destination / "notes.txt", "w") as file:
                file.write(self.notes)