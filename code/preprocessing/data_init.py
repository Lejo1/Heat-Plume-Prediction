from torch.utils.data import DataLoader

from preprocessing.datasets.dataset import DataPoint
from preprocessing.datasets.dataset_cuts_jit import SimulationDatasetCuts

def init_data(args:dict, tmp_bool_cutouts:bool=False, batchsize:int=64, ORDER_DATA:list=[0,2,1]):
    dataloaders = {}
    
    if not tmp_bool_cutouts or args["case"] == "test": # NO CUTOUTS
        dataset_train = DataPoint(args["data_prep"], i=ORDER_DATA[0])
        
    else: # DO CUTOUTS
        dataset_train = SimulationDatasetCuts(args["data_prep"], skip_per_dir=args["skip_per_dir"], box_size=args["len_box"], ids=ORDER_DATA[0])
    dataset_val = DataPoint(args["data_prep"], i=ORDER_DATA[1])
    datasets = [dataset_train, dataset_val]

    dataloaders["train"] = DataLoader(datasets[0], batch_size=batchsize, shuffle=True, num_workers=0)
    dataloaders["val"] = DataLoader(datasets[1], batch_size=batchsize, shuffle=True, num_workers=0)
    try:
        dataset_test = DataPoint(args["data_prep"], i=ORDER_DATA[2])
        datasets.append(dataset_test)
        dataloaders["test"] = DataLoader(datasets[2], batch_size=batchsize, shuffle=False, num_workers=0)
    except:
        pass

    return datasets[0].input_channels, datasets[0].output_channels, dataloaders