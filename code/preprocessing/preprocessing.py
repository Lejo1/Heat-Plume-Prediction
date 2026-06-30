from pathlib import Path
from utils.utils_args import is_empty, load_yaml, save_yaml

def preprocessing(args:dict):
    assert not is_unprepared(args["data_prep"]), f"For benchmark case I currently expect the data to be prepared!"
    print(f"Dataset: {args['data_prep']}")

    if args["case"] == "train":
        info = load_yaml(args["data_prep"]/"info.yaml") 
        save_yaml(info, args["destination"]/"info.yaml")

# helper function
def is_unprepared(path:Path):
    return is_empty(path / "Inputs") or is_empty(path / "Labels") or not (path / "info.yaml").exists()