import matplotlib.pyplot as plt
import torch

from preprocessing.datasets.dataset_extend import DatasetExtend
from processing.networks.unetVariants import UNetHalfPad2
from utils.utils_args import load_yaml


def extend_plumes_inference_pipeline(run_id, path_data, path_sparnn, path_1hpnn=None, use_prediction:bool=False, verbose:bool=True):
    # load data
    hparams = load_yaml(path_sparnn / "HPS_options.yaml")
    len_box = hparams["len_box"]["values"][0] #224
    dataset = DatasetExtend(path=path_data, kernel_size=hparams["kernel_size"]["values"][0], n_blocks=hparams["depth"]["values"][0], box_size=len_box)
    first_box_id = run_id * dataset.dp_per_run  # choose run id
    front = dataset.front_exclude
    width= dataset[0][0].shape[-1]

    # init overall temperature fields
    temperature_prediction = torch.zeros((len_box+front, width))
    temperature_label = torch.zeros((len_box+front, width))
    if path_1hpnn is None:
        temperature_prediction[:front] = torch.load(path_data / "Labels" / dataset.input_names[run_id])[0,:front]
        temperature_prediction = dataset.insert_in_domain(0, temperature_prediction, dataset[first_box_id][0][-1], case="Inputs")
        temperature_label[:len_box+front] = torch.load(path_data / "Labels" / dataset.input_names[run_id])[0,:len_box+front]
    else:
        raise NotImplementedError("1HP-NN inference not implemented yet.")
        # load 1HP-NN, TMP: current not existing -> use label
        # apply 1HP-NN (and store results in a field called temperature_prediction)

    if verbose:
        plt.imshow(temperature_prediction.numpy().T)
        plt.title('Initial Prediction')
        plt.colorbar(label='Normalized Temperature')
        plt.show()

    # load SpaR-NN
    model = UNetHalfPad2(in_channels=3, out_channels=1, init_features=hparams["init_features"]["values"][0], depth=hparams["depth"]["values"][0], kernel_size=hparams["kernel_size"]["values"][0], activation=hparams["activation_fct"]["values"][0], norm=hparams["norm"]["values"][0])
    model.load(path_sparnn)
    model.eval()

    ## iteration:
    max_iterations = 20
    i = 0
    while criterion_iteration(temperature_prediction, i, max_iterations=max_iterations):
        # apply SpaR-NN while criterion_iteration is True (and store results, field is extended in each iteration)
        tmp_inputs, tmp_labels = dataset[first_box_id + i]
        if use_prediction:
            start = dataset.get_start_loc(i)
            tmp_inputs[-1] = temperature_prediction[start : start + len_box]

        out = model(tmp_inputs.unsqueeze(0)).squeeze().detach()
        temperature_prediction = dataset.insert_in_domain(i, temperature_prediction, out, case="Outputs")
        temperature_label = dataset.insert_in_domain(i, temperature_label, tmp_labels[0], case="Labels") # used as stopping criterion in case we exceed the cells in the original data

        # visualize results
        if verbose:
            plt.figure(figsize=(10,2))
            plt.imshow((temperature_prediction).numpy().T)
            # plt.imshow((temperature_prediction-temperature_label[:-dataset.gap]).numpy().T)
            plt.title(f'Iteration {i}')
            plt.colorbar(label='Normalized Temperature')
            plt.show()
        
        i += 1

    return temperature_prediction, temperature_label


def criterion_iteration(temperature_prediction, i, max_iterations=10):
    temperature_reference = 0.0 # background temperature is 10.6 degree but normed to 0.0, expecting the temperature injection to be positive
    epsilon = 2 * 1e-2 # should correspond to 0.1 degree Celsius
    return (torch.abs(temperature_prediction[-1,:] - temperature_reference) > epsilon).any() and (i < max_iterations)