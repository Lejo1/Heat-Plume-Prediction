from pathlib import Path
import yaml
from typing import List, Union
from torch import Tensor
import numpy as np

def read_cla(path:str):
    clas = load_yaml(path / "command_line_arguments.yaml")
    for path_typed_cla in ["data_prep", "data_raw", "model", "destination"]:
        try:
            if clas[path_typed_cla] is not None:
                clas[path_typed_cla] = Path(clas[path_typed_cla])
        except KeyError:
            continue
    clas["destination"] = path

    return clas
    
def make_data_prep_dir(args:dict, prep_dir: Path=None):
    print(f"Dataset_pre path: {args['data_prep']}")
    if args["data_prep"] is None:
        args["data_prep"] = args["data_raw"].name + " inputs_" + args["inputs"] + " outputs_" + args["outputs"]
        args["data_prep"] = prep_dir / args["data_prep"]

    args["data_prep"].mkdir(parents=True, exist_ok=True)
    (args["data_prep"] / "Inputs").mkdir(parents=True, exist_ok=True)
    (args["data_prep"] / "Labels").mkdir(parents=True, exist_ok=True)
    save_yaml(args, args["destination"] / "command_line_arguments.yaml")

def check_model_avail(args:dict):
    # model, destination
    if not (args["destination"] / "model.pt").exists() or not (args["destination"] / "info.yaml").exists() or not (args["destination"] / "HPS_options.yaml").exists():
        raise FileNotFoundError(f"model.pt or info.yaml or HPS_options.yaml not found in {args['destination']}")

def load_yaml(path: Path, **kwargs) -> dict:
    with open(path, "r") as file:
        args = yaml.safe_load(file, **kwargs)
    return args

# Convert tensors to Python-native types
def convert_to_python_datatypes(data):
    if isinstance(data, Tensor):
        return data.item() if data.numel() == 1 else data.tolist()
    elif isinstance(data, (np.ndarray, np.generic)):
        return data.item() if np.isscalar(data) else data.tolist()
    elif isinstance(data, dict):
        return {k: convert_to_python_datatypes(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_python_datatypes(v) for v in data]
    else:
        return data


def save_yaml(args:dict, destination_file):
    with open(destination_file, "w") as file:
        tmp = args.copy()
        for arg in args.keys():
            try:
                for info in arg.keys():
                    tmp[info] = path_to_str(arg[info])
            except:
                tmp[arg] = path_to_str(args[arg])
        # Convert tensors to Python-native types
        tmp = convert_to_python_datatypes(tmp)
        # Save to YAML file
        yaml.dump(tmp, file, default_flow_style=False)

def path_to_str(arg: Union[Path, str]) -> str:
    '''if arg a Path object, convert to string'''
    if isinstance(arg, Path):
        return str(arg)
    return arg

def get_run_ids_from_prep(dir: Path) -> List[int]:
    run_ids = []
    for file in dir.iterdir():
        if file.suffix == ".pt":
            run_ids.append(int(file.stem.split("_")[-1]))
            # print(f"Found run_id {run_ids[-1]}")
    run_ids.sort()
    return run_ids

# OTHER UTILS
def is_empty(path:Path):
    return not bool(list(path.iterdir()))