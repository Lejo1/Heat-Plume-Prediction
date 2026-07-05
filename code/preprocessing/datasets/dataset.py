import os
import pathlib
import torch
import yaml
from torch.utils.data import Dataset

from preprocessing.transforms import NormalizeTransform
from utils.utils_args import get_run_ids_from_prep

class DatasetBasis(Dataset):
    def __init__(self, path:str, box_size:int=None, idx:int=None):
        Dataset.__init__(self)
        self.path = pathlib.Path(path)
        self.info = self.__load_info()
        self.norm = NormalizeTransform(self.info)
        self.input_names = []
        self.label_names = []
        for filename in os.listdir(self.path / "Inputs"):
            self.input_names.append(filename)
        for filename in os.listdir(self.path / "Labels"):
            self.label_names.append(filename)
        self.input_names.sort()
        self.label_names.sort()
        self.spatial_size = torch.load(self.path / "Labels" / self.input_names[0]).shape[1:]
        if box_size is not None:
            self.box_size = box_size
        else:
            self.box_size = self.spatial_size[0]

        if len(self.input_names) != len(self.label_names):
            raise ValueError(
                "Number of Inputs and labels does not match!")

    @property
    def input_channels(self):
        return len(self.info["Inputs"])

    @property
    def output_channels(self):
        return len(self.info["Labels"])

    def __load_info(self):
        with open(self.path / "info.yaml", "r") as f:
            info = yaml.safe_load(f)
        return info

    def __len__(self):
        return len(self.input_names)
    
    def __getitem__(self, idx):
        input = torch.load(self.path / "Inputs" / self.input_names[idx])[:, :self.box_size, :]
        label = torch.load(self.path / "Labels" / self.label_names[idx])[:, :self.box_size, :]
        return input, label

class DataPoint(DatasetBasis):
    def __init__(self, path:str, i:int=0):
        DatasetBasis.__init__(self, path)
        if isinstance(i, int):
            run_id = get_run_ids_from_prep(self.path / "Inputs")[i]
            
            self.input_names = [f"RUN_{run_id}.pt"]
            self.label_names = [f"RUN_{run_id}.pt"]
        elif isinstance(i, list):
            self.input_names = [f"RUN_{get_run_ids_from_prep(self.path / 'Inputs')[ii]}.pt" for ii in i]
            self.label_names = [f"RUN_{get_run_ids_from_prep(self.path / 'Labels')[ii]}.pt" for ii in i]
        else:
            raise ValueError("i must be an int or a list of ints")
        self.input_names.sort()
        self.label_names.sort()

class DataPointE2E(Dataset):
    """Full-domain datapoint for end-to-end LGCNN training: joins two prepared datasets.

    x: normalized pki inputs from `<dataset> inputs_pki outputs_xy`,
    y: 3 channels [T, vx, vy] - normalized T label from `<dataset> inputs_ixyk outputs_t for_s`
       plus the normalized simulated velocities from the pki dataset's Labels (used by the
       auxiliary velocity loss).
    Both prep dirs are produced by LGCNN_step2.py (STEP 1); their normalization stats for the
    shared channels (i, k, vx, vy) are identical by construction.
    """
    input_channels = 3   # p, k, i
    output_channels = 3  # T + vx + vy

    def __init__(self, prep_dir:pathlib.Path, dataset_name:str, i:int=0):
        Dataset.__init__(self)
        self.path_v = pathlib.Path(prep_dir) / f"{dataset_name} inputs_pki outputs_xy"
        self.path_T = pathlib.Path(prep_dir) / f"{dataset_name} inputs_ixyk outputs_t for_s"
        for path in [self.path_v, self.path_T]:
            assert path.exists(), f"{path} not found - run the data preparation (STEP 1) of LGCNN_step2.py first"
        with open(self.path_v / "info.yaml", "r") as f:
            self.info_v = yaml.safe_load(f)
        with open(self.path_T / "info.yaml", "r") as f:
            self.info_T = yaml.safe_load(f)
        self.norm_T = NormalizeTransform(self.info_T)  # for de-normalizing T in plots/inference

        run_id = get_run_ids_from_prep(self.path_v / "Inputs")[i]
        self.input_names = [f"RUN_{run_id}.pt"]
        self.label_names = [f"RUN_{run_id}.pt"]

    def __len__(self):
        return len(self.input_names)

    def __getitem__(self, idx):
        input = torch.load(self.path_v / "Inputs" / self.input_names[idx])
        label_T = torch.load(self.path_T / "Labels" / self.label_names[idx])
        label_v = torch.load(self.path_v / "Labels" / self.label_names[idx])
        return input, torch.cat([label_T, label_v])