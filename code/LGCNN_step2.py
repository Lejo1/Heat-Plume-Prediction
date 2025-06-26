from pathlib import Path

from utils.utils_args import make_data_prep_dir
from preprocessing.preprocessing import preprocessing
from step2_streamlines.streamlines_main import build_streamlines

# this script runs the following steps:
# 1. prepare data
# 2. calculate streamlines with simulated or with predicted velocity fields

PATH_DATA_RAW = Path("../datasets") # TODO: change to your path
PATH_DATA_PREP = Path("../datasets_prep") # TODO: change to your path
PATH_MODELS_DIR = Path("../runs") # TODO: change to your path

if __name__ == "__main__":
    # for data preparation
    dataset_name = "dataset_giant_100hp_varyK"
    randomK = True #TODO True if randomK dataset vs. False if realK dataset; because main-direction of flow is switched

    # for streamlines, partially only required if based_on_pred is True
    method = "RK45" # or RK23 or Radau # TODO adapt to your needs
    based_on_pred = True # based on simulated or predicted velocities # TODO adapt to your needs
    model = PATH_MODELS_DIR / "MODEL_STEP1" # TODO adapt to your needs


    # STEP 1: prepare data (ixyk->T and pki->xy)
    args = {
        "inputs": "ixyk",
        "outputs": "t",
        "case": "train",
        "model": None,
        "destination": PATH_MODELS_DIR / "tmp", # dummy
        "data_raw" : PATH_DATA_RAW / dataset_name,
        "data_prep" : PATH_DATA_PREP / f"{dataset_name} inputs_ixyk outputs_t for_s",
    }
    args["destination"].mkdir(parents=True, exist_ok=True) # dummy
    make_data_prep_dir(args)
    preprocessing(args) # and save info.yaml in model folder

    args["inputs"] = "pki"
    args["outputs"] = "xy"
    args["data_prep"] = None
    make_data_prep_dir(args, prep_dir=PATH_DATA_PREP)
    preprocessing(args) # and save info.yaml in model folder

    # STEP 2: calculate streamlines with simulated and with predicted velocity fields
    streamlines = build_streamlines(model, PATH_DATA_PREP / dataset_name, based_on_pred=based_on_pred, method=method, randomK=randomK)