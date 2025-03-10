from data_stuff.dataset import SimulationDataset, get_splits
from torch.utils.data import DataLoader, random_split
from data_stuff.utils import SettingsTraining
import torch

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