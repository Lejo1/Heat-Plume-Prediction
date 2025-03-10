from dataclasses import dataclass
import os
import pathlib
import typing
import yaml

# Data classes for paths
@dataclass
class Paths2HP:
    raw_path: pathlib.Path # domain
    dataset_model_trained_with_prep_path: pathlib.Path # 1hp-boxes
    dataset_1st_prep_path: pathlib.Path # domain
    model_1hp_path: pathlib.Path
    datasets_boxes_prep_path: pathlib.Path # 2hp-boxes

# Functions for setting paths
def set_paths_2hpnn(dataset_name: str, preparation_case: str, model_name: str = None, dataset_prep:str = None, already_prep:bool = False, paths_file:str = "paths.yaml")-> typing.Tuple[Paths2HP, str, pathlib.Path]:
    
    if not os.path.exists(paths_file):
        raise FileNotFoundError(f"{paths_file} not found")
    with open(paths_file, "r") as f:
        paths = yaml.safe_load(f)

    datasets_raw_domain_dir = pathlib.Path(paths["default_raw_dir"])
    datasets_prepared_domain_dir = pathlib.Path(paths["datasets_prepared_dir"])
    prepared_1hp_dir = pathlib.Path(paths["destination_dir"])
    destination_dir = pathlib.Path(paths["destination_dir"])
    datasets_prepared_2hp_dir = pathlib.Path(paths["generated_dataset_dir"])

    prepared_1hp_dir = prepared_1hp_dir / preparation_case
    if not model_name:
        for path in prepared_1hp_dir.iterdir():
            if path.is_dir():
                if "current" in path.name: # TODO change to "model"
                    model_1hp_path = prepared_1hp_dir / path.name
                elif "dataset" in path.name:
                    dataset_model_trained_with_prep_path = prepared_1hp_dir / path.name
    else:
        model_1hp_path = pathlib.Path(paths["destination_dir"]) / model_name
        dataset_model_trained_with_prep_path = model_1hp_path
    
    dataset_raw_path = datasets_raw_domain_dir / dataset_name
    inputs = preparation_case
    dataset_1st_prep_path = datasets_prepared_domain_dir / f"{dataset_name} inputs_{inputs}"
    if dataset_prep == "" and not already_prep:
        dataset_prep_2hp_path = f"{dataset_name} inputs_{preparation_case} boxes"
        dataset_1st_prep_path = datasets_prepared_domain_dir / f"{dataset_name} inputs_{inputs}"
    else:
        dataset_prep_2hp_path = dataset_prep
        dataset_1st_prep_path = datasets_prepared_domain_dir / dataset_prep
    datasets_boxes_prep_path = datasets_prepared_2hp_dir / dataset_prep_2hp_path

    return Paths2HP(
        dataset_raw_path,
        dataset_model_trained_with_prep_path,
        dataset_1st_prep_path,
        model_1hp_path,
        datasets_boxes_prep_path,
        ), inputs, destination_dir

def extend_paths_for_architecture(architecture:str, default_raw_dir: pathlib.Path, destination_dir: pathlib.Path, datasets_prepared_dir: pathlib.Path)-> typing.Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    if architecture in ["extend1", "extend2"]:
        default_raw_dir = default_raw_dir / "extend_plumes"
        datasets_prepared_dir = datasets_prepared_dir / "extend_plumes"
        if architecture == "extend1":
            destination_dir = destination_dir / "extend_plumes1"
        else:
            destination_dir = destination_dir / "extend_plumes2"
    elif architecture == "allin1":
        default_raw_dir = default_raw_dir / "giant_manyhps"
        destination_dir = destination_dir / "allin1"
        datasets_prepared_dir = datasets_prepared_dir / "giant_manyhps"
    elif architecture in ["standard","parallel","quad"]:
        default_raw_dir = default_raw_dir
        destination_dir = destination_dir
        datasets_prepared_dir = datasets_prepared_dir
    else:
        raise ValueError(f"architecture {architecture} not known")
    return default_raw_dir, destination_dir, datasets_prepared_dir