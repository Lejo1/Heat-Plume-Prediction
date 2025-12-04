import torch
from torch.utils.data import DataLoader, random_split

from preprocessing.datasets.dataset import DataPoint
from preprocessing.datasets.dataset_1stbox import Dataset1stBox
from preprocessing.datasets.dataset_cuts_jit import SimulationDatasetCuts
from preprocessing.datasets.dataset_extend import DatasetExtend, random_split_extend

def init_data(args:dict, seed=1):
    batchsize:int = args["batchsize"] if args["batchsize"] else 64
    tmp_bool_cutouts: bool = args["bool_cutouts"]
    order_data:list = args["order_data"] if args["order_data"] else None
    kernel_size:int = args["kernel_size"] if args["kernel_size"] else None
    n_blocks=args["depth"]
    n_convs_per_block=1
    box_size=args["len_box"]

    if args["problem"] == "allin1":
        print("order", order_data)
        dataloaders = {}
        
        if not tmp_bool_cutouts or args["case"] == "test": # NO CUTOUTS
            dataset_train = DataPoint(args["data_prep"], i=order_data[0])
        else: # DO CUTOUTS
            dataset_train = SimulationDatasetCuts(args["data_prep"], skip_per_dir=args["skip_per_dir"], box_size=args["len_box"], ids=order_data[0])
        dataset_val = DataPoint(args["data_prep"], i=order_data[1])
        datasets = [dataset_train, dataset_val]

        dataloaders["train"] = DataLoader(datasets[0], batch_size=batchsize, shuffle=True, num_workers=0)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=batchsize, shuffle=True, num_workers=0)
        try:
            dataset_test = DataPoint(args["data_prep"], i=order_data[2])
            # dataset_val = SimulationDatasetCuts(args["data_prep"], skip_per_dir=args["skip_per_dir"], box_size=args["len_box"], idx=[0,])
            datasets.append(dataset_test)
            dataloaders["test"] = DataLoader(datasets[2], batch_size=batchsize, shuffle=False, num_workers=0)
        except:
            pass
        print(f"Length of datasets: {len(dataset_train)}:{len(dataset_val)}, with {datasets[0].input_channels} input channels and {datasets[0].output_channels} output channels")

        return datasets[0].input_channels, datasets[0].output_channels, dataloaders
        
    # ALL OTHER CASES THAN ALLIN1 don't need order
    else:
        if args["problem"] in ["2stages", "1hp", "test"]:
            dataset = Dataset1stBox(args["data_prep"], box_size=args["len_box"])
        elif args["problem"] == "extend":
            dataset = DatasetExtend(args["data_prep"], kernel_size=kernel_size, n_blocks=n_blocks, n_convs_per_block=n_convs_per_block, box_size=box_size)
            args["inputs"] += "T"

        split_ratios = [0.7, 0.2, 0.1]
        generator = torch.Generator().manual_seed(seed)
        if args["problem"] in ["2stages", "1hp", "test"]:
            datasets = random_split(dataset, split_ratios, generator=generator)
        elif args["problem"] == "extend":
            datasets = random_split_extend(dataset, split_ratios, generator=generator)

        dataloaders = {}
        try:
            dataloaders["train"] = DataLoader(datasets[0], batch_size=batchsize, shuffle=True, num_workers=0)
            dataloaders["val"] = DataLoader(datasets[1], batch_size=batchsize, shuffle=True, num_workers=0)
        except: pass
        dataloaders["test"] = DataLoader(datasets[2], batch_size=batchsize, shuffle=False, num_workers=0)
        print(f"Length of dataset: {len(dataset)} - split into {len(datasets[0])}:{len(datasets[1])}:{len(datasets[2])}")
        return dataset.input_channels, dataset.output_channels, dataloaders

def load_all_datasets_in_full(args: dict, ORDER_DATA: list = [0, 2, 1]):
    dataloaders = {}
    for i, case in zip(ORDER_DATA, ["train", "val", "test"]):
        if args["problem"] == "allin1":
            try:
                dataset = DataPoint(args["data_prep"], i=i)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
                dataloaders[case] = dataloader
            except: pass
        else:
            dataset = Dataset1stBox(args["data_prep"])
            dataloader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=0)
            dataloaders[case] = dataloader
    return dataloaders    