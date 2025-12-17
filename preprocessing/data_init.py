import torch
from torch.utils.data import DataLoader, random_split

from preprocessing.datasets.dataset import DataPoint
from preprocessing.datasets.dataset_1stbox import Dataset1stBox
from preprocessing.datasets.dataset_cuts_jit import SimulationDatasetCuts
from preprocessing.datasets.dataset_extend import DatasetExtend, random_split_extend

def init_data(args:dict, seed=1):
    batchsize = args["batchsize"] if "batchsize" in args else 64
    box_size = args["len_box"] if "len_box" in args else None

    if args["problem"] == "allin1":
        ORDER_DATA = args["order_data"] if "order_data" in args else [0,2,1]
        tmp_bool_cutouts = args["bool_cutouts"] if "bool_cutouts" in args else False
        print("order", ORDER_DATA)

        dataloaders = {}
        
        if not tmp_bool_cutouts or args["case"] == "test": # NO CUTOUTS
            dataset_train = DataPoint(args["data_prep"], i=ORDER_DATA[0])
        else: # DO CUTOUTS
            dataset_train = SimulationDatasetCuts(args["data_prep"], skip_per_dir=args["skip_per_dir"], box_size=box_size, ids=ORDER_DATA[0])
        dataset_val = DataPoint(args["data_prep"], i=ORDER_DATA[1])
        datasets = [dataset_train, dataset_val]

        dataloaders["train"] = DataLoader(datasets[0], batch_size=batchsize, shuffle=True, num_workers=0)
        dataloaders["val"] = DataLoader(datasets[1], batch_size=batchsize, shuffle=True, num_workers=0)
        try:
            dataset_test = DataPoint(args["data_prep"], i=ORDER_DATA[2])
            # dataset_val = SimulationDatasetCuts(args["data_prep"], skip_per_dir=args["skip_per_dir"], box_size=box_size, idx=[0,])
            datasets.append(dataset_test)
            dataloaders["test"] = DataLoader(datasets[2], batch_size=batchsize, shuffle=False, num_workers=0)
        except:
            pass
        print(f"Length of datasets: {len(dataset_train)}:{len(dataset_val)}, with {datasets[0].input_channels} input channels and {datasets[0].output_channels} output channels")

        return datasets[0].input_channels, datasets[0].output_channels, dataloaders
        
    # ALL OTHER CASES THAN ALLIN1 don't need order
    else:
        if args["problem"] in ["2stages", "1hp", "test"]:
            dataset = Dataset1stBox(args["data_prep"], box_size=box_size)
        elif args["problem"] == "extend":
            kernel_size = args["kernel_size"] if "kernel_size" in args else None
            n_blocks = args["depth"] if "depth" in args else None
            n_convs_per_block = args["n_convs_per_block"] if "n_convs_per_block" in args else 1

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
            dataloaders["train"] = DataLoader(datasets[0], batch_size=batchsize, shuffle=True, 
            num_workers=4,       # parallel indexing
            pin_memory=True,     # faster CPU → GPU transfer
            persistent_workers=True)
            dataloaders["val"] = DataLoader(datasets[1], batch_size=batchsize, shuffle=True, 
            num_workers=4,       # parallel indexing
            pin_memory=True,     # faster CPU → GPU transfer
            persistent_workers=True)
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