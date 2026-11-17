# Training, Hyperparameter Search and Evaluation of LGCNN with End-to-End training
This repository contains code for hyperparameter optimization and training of LGCNN and a vanilla UNet model using PyTorch. It also includes functionality for evaluating the trained model and reproducing results from the associated research.

It is based on the original code of Pelzer et al., https://github.com/JuliaPelzer/Heat-Plume-Prediction/tree/AllIn1/LGCNN/release25, and extends it by an end-to-end training of the LGCNN: the streamline calculation (Step 2) is differentiable, so that CNN1 (Step 1) and CNN2 (Step 3) can be trained jointly on the temperature loss, optionally with additional losses on the streamlines.

Pre-trained models and raw datasets are part of the supplementary material of the research. The preparation of the datasets for the first step or the full pipeline is done automatically, when `main.py` is run and the datasets is not yet prepared or not in the correct folder (see below). For Step 3, the data has to be prepared manually, see Step 2.
The trained models can be found at: https://doi.org/10.18419/DARUS-6683

## Table of Contents
1. Some important files
2. Getting started
3. Training of an LGCNN (or another model)
4. End-to-end training of an LGCNN
5. Inference of an LGCNN
6. Postprocessing, incl. evaluation of metrics
7. Results

## Some important files in code/
- **`main.py`**: Contains code for training a single model (Step 1 and 3 of LGCNN, and vanilla UNet_3dp), for the end-to-end training of LGCNN, or conducting a hyperparameter search.
- **`LGCNN_step2.py`**: Contains code for calculating streamlines (Step 2) and preparing inputs for Step 3.
- **`processing/training_e2e.py`**, **`processing/networks/lgcnn_e2e.py`**: End-to-end training and model (CNN1 -> differentiable streamlines -> CNN2).
- **`step2_streamlines/streamlines_helpers.py`**: Differentiable streamline tracing (RK4) and soft drawing of the streamlines.
- **`processing/loss_fcts.py`**: Loss functions, incl. the trajectory loss and the streamline-frame losses ($L_N$, $L_{TOF}$).
- **`eval_metrics.py`**: Computes metrics for the train, test, validation, and scaling datasets using a given model.
- **`eval_scaling_e2e.py`**: Computes the metrics of an end-to-end model on the scaling, validation or test dataset.
- **`postprocessing/thesis/`**, **`postprocessing/presentation/`**: Scripts for the figures and tables of the thesis and the figures of the final presentation.
- **`default_HPS_options.yaml`**: Defines the default search ranges for hyperparameter optimization. Needs to be copied to the hyperparameter search folder and renamed to `HPS_options.yaml` to be used, also used for setting the model parameters if no hyperparameter search is performed.
- **`runs/example`**: Contains example files for the command line arguments and hyperparameters. These files need to be copied to the folder where a new model is trained.
- **`runs/`**: The other folders contain the configurations of the end-to-end runs of the results (see Results) and of the hyperparameter searches.

## Getting started
- install the requirements: via`pip install -r requirements.txt`
- download the raw datasets, optional: trained models
- make separate folders for the raw and prepared datasets and the models, e.g., `datasets`, `datasets_prep` and `runs` as in the example
- `cd` into the code folder

## Training of LGCNN (or another model)
Check the path variables

    PATH_DATA_RAW = Path("../datasets") # TODO: change to your path
    PATH_DATA_PREP = Path("../datasets_prep") # TODO: change to your path
    PATH_MODELS_DIR = Path("../runs") # TODO: change to your path

in `main.py` and `LGCNN_step2.py`.

### Step 1 (predict v)
- make a new folder for the model in `runs/`, e.g., `NAME_OF_DIR_PREDICT_V`
- define a `command_line_argument.yaml` in `NAME_OF_DIR_PREDICT_V` acc. to the example in `example/command_line_arguments_v.yaml`, and `HPS_options.yaml` in `NAME_OF_DIR_PREDICT_V` acc. to the example in `example/HPS_options.yaml` (hyperparameters of your model or of the hyperparameter search)
- run `python main.py --destination NAME_OF_DIR_PREDICT_V`
- if you want to run a hyperparameter search, make sure to add the parameter `--hsearch True` and to give more than one value for the hyperparameters in `HPS_options.yaml`

###  Step 2 (calculate streamlines and prepare inputs for Step 3)
- open `LGCNN_step2.py` and set 
    - the path variables, 
    - the `dataset_name`, 
    - whether it is the synthetic-permeability(randomK)-dataset or the realK one (`randomK=True/False`), 
    - if the streamline calculation should be based on the predicted velocity field or the original velocity field (`based_on_pred=True/False`). If this is set to `True`, you also need a model-path in `model`
    - the streamlines are calculated with a fixed-step Runge-Kutta 4 method; `method` only names the prepared dataset folder
- run `python LGCNN_step2.py`

### Step 3 (predict T)
- check that the required prepared dataset exists (download or run Step 2)
- define a `command_line_argument.yaml` in `NAME_OF_DIR_PREDICT_T` acc. to example in `example/command_line_arguments_T.yaml` and `HPS_options.yaml` (same as in Step 1)
- make sure that `data_pred`in `command_line_arguments.yaml` points to the prepared dataset from Step 2, i.e., that the path is correct, e.g., `datasets_prep/dataset_100hp_giant_real_fixP0_0025 inputs_ixydk+s_outer outputs_t`
- run `python main.py --destination NAME_OF_DIR_PREDICT_T`
- if you want to run a hyperparameter search, make sure to add the parameter `--hsearch True` 

## End-to-end training of LGCNN
- prepare the datasets with `LGCNN_step2.py` (its first part prepares `<dataset> inputs_pki outputs_xy` and `<dataset> inputs_ixyk outputs_t for_s`, which the end-to-end training reads)
- `pipeline: e2e` in `command_line_arguments.yaml` selects the end-to-end training; the folders in `runs/` (e.g., `runs/joint_frame_losses`) contain complete configurations, the options are explained in the comments of `runs/finetune_e2e/command_line_arguments.yaml`
    - `case: finetune`: CNN1 and CNN2 are loaded from two trained models of Step 1 and Step 3 (`model_v`, `model_T`, e.g., the paper's models `BEST_predict_v_v4` and `BEST_predict_T_add_s_outer`)
    - `case: train`: CNN2 is trained from scratch; CNN1 is loaded from `model_v` or pretrained on the velocities first (`stage1_epochs`)
- run `python main.py --destination NAME_OF_DIR_E2E`, e.g., `python main.py --destination joint_frame_losses`
- `runs/frame_CNN1_CNN2_retrained` needs the CNN1 of `runs/frame_CNN1_alone`, see the comment in its `command_line_arguments.yaml`
- hyperparameter search: `python main.py --destination hsearch_e2e --hsearch True` (finetuning) or `--destination hsearch_example_e2e` (from scratch)
- quick checks on the CPU: `python test_e2e_training.py`, `python test_frame_loss.py`

### Options of the end-to-end training (`command_line_arguments.yaml`)
The loss types `train_loss` and `v_loss` and the learning rate `lr` are set in `HPS_options.yaml`.

| Option | Default | Description |
|---|---|---|
| `pipeline` | - | `e2e` selects the end-to-end training |
| `case` | - | `finetune`, `train` or `test` (see above) |
| `model_v`, `model_T` / `model` | - | the two trained models of Step 1 and Step 3 / one end-to-end checkpoint |
| `unet_args_T` | - | architecture keys of CNN2 that differ from CNN1 (e.g., `kernel_size: 4`) |
| `stage1_epochs`, `stage1_lr_schedule`, `stage1_force` | 0, -, false | velocity pretraining of CNN1 (case `train` without `model_v`); `stage1_force` retrains a cached CNN1 |
| `lr_stage2`, `lr_schedule` | `lr`, - | learning rate of the end-to-end training, schedule `{epoch: lr}` |
| `clip_grad` | 1.0 | cap on the global gradient norm |
| `lambda_v` | 0 | weight of the velocity loss (type `v_loss`) |
| `val_loss`, `val_loss_with_v` | training loss, true | validation loss (selects the best epoch); `false` drops the velocity term |
| `lambda_traj` | 0 | weight of the trajectory loss (distance of the streamlines of CNN1 to the simulated ones) |
| `lambda_N`, `lambda_TOF` | 0, 0 | weights of the streamline-frame losses (direction, travel time) |
| `seed_mode`, `n_seeds` | hp, 200 | start points of the frame-loss streamlines: heat pumps, random, or both |
| `frame_eps`, `tof_floor`, `tof_checkpoint_cells`, `well_mask_*`, ... | see `FrameLoss.DEFAULTS` | further options of the frame losses (`processing/loss_fcts.py`) |
| `log_frame_metrics` | false | log $L_N$, $L_{TOF}$ and the angle error every epoch |
| `t_steps`, `sigma`, `fade_mode` | -, -, absolute | streamline samples, width of the soft drawing in cells (< 1.5), fading of the lines |
| `v_blur`, `v_blur_end` | 0, `v_blur` | Gaussian smoothing of the velocity read by the streamline tracer, annealed linearly to `v_blur_end` |
| `detach_direct_v` | false | CNN1 gets the temperature gradient only through the streamlines |
| `detach_trajectory` | false | truncated gradient through the streamlines (each RK4 step only) |
| `compile` | false | `torch.compile` the RK4 step |
| `freeze_weights` | none | `v` / `T`: weights of CNN1 / CNN2 stay fixed |
| `cnn1_trainable` | all | `bn`, `last`, `bn_last`: train only the BatchNorm and/or the last layer of CNN1 |
| `freeze_bn` | none | `v`, `T`, `both`: BatchNorm statistics of CNN1 / CNN2 stay fixed |
| `bn_reestimate` | false | recompute the BatchNorm statistics on the full domain before the training |
| `visualize`, `grad_diag`, `grad_diag_only`, `pipeline_plot_every`, `pipeline_plot_max` | false, false, false, 0, 0 | plots of the start and end state, gradient diagnostic, plots of every stage during the training |
| `n_trials`, `pruner_startup_trials`, `pruner_warmup_epochs` | 50, 3, 10 | hyperparameter search (`--hsearch True`) |
| `subdomain` | false | train and evaluate on one fixed 736 x 736 window (debugging) |

## Inference of LGCNN
- make sure, that you have a trained model for Step 1 and Step 3, e.g., `NAME_OF_DIR_PREDICT_V` and `NAME_OF_DIR_PREDICT_T`
- prepare a new dataset for Step 2 based on the outputs of Step 1 and the predicted velocities (see parameters in `LGCNN_step2.py`: `based_on_pred=True` and `model=NAME_OF_DIR_PREDICT_V`)
- in `NAME_OF_DIR_PREDICT_T`, set `case: test` in `command_line_arguments.yaml` to only run inference and set `visu: True` to visualize the results
- run `python main.py --destination NAME_OF_DIR_PREDICT_T`

End-to-end model (one forward pass pki -> T, needs `model.pt` in the run folder):
- run `python LGCNN_e2e_inference.py --run NAME_OF_DIR_E2E [--datapoint 2] [--device cuda]`

## Postprocessing
- set `visu: True` during training / inference in `command_line_arguments.yaml`
- set `case: test` in `command_line_arguments.yaml` to only run inference

### Evaluation
- run `eval_metrics.py` with a proper dataset and model paths to evaluate metrics as in the paper, e.g., `python eval_metrics.py --model ../runs/NAME_OF_DIR_PREDICT_T --data "../datasets_prep/<dataset> inputs_ixydk+s_outer outputs_t prep_with_<model> RK45"` (`--scaling` for the scaling dataset)
- end-to-end models: the test metrics are written to `measurements_test.yaml` at the end of the training; `python eval_scaling_e2e.py --run NAME_OF_DIR_E2E [--split scaling|val|test]` evaluates the scaling, validation or test dataset
- the tables and figures of the thesis are produced by the scripts in `postprocessing/thesis/`

## Results
This section contains a summary of the results of the end-to-end training (bachelor thesis). The LGCNN is trained on datapoint 0, validated on datapoint 1 and tested on datapoint 2 of `dataset_giant_100hp_varyK`; the scaling test uses `giant_double_100hp_v2`. MAE of the temperature in °C, computed with `evaluate_e2e` (end-to-end) and `eval_metrics.py` (two-step models).

| Configuration                    | Run folder in `runs/`         | val MAE | test MAE | scaling MAE |
|----------------------------------|-------------------------------|---------|----------|-------------|
| LGCNN (Pelzer et al.)            | paper models                  | 0.0966  | 0.0902   | 0.0413      |
| soft streamlines, untrained      | start of the finetuning runs  | 0.0932  | 0.0864   | 0.0386      |
| sequential CNN2                  | `sequential_CNN2`             | 0.0943  | 0.0860   | 0.0394      |
| CNN2 alone                       | `CNN2_alone`                  | 0.0902  | 0.0828   | 0.0365      |
| CNN1 alone through streamlines   | `CNN1_through_streamlines`    | 0.0909  | 0.0901   | 0.0383      |
| joint T loss                     | `joint_T_loss`                | 0.0907  | 0.0907   | 0.0387      |
| joint + frame losses             | `joint_frame_losses`          | 0.0884  | 0.0857   | 0.0348      |
| frame-loss CNN1 + CNN2 retrained | `frame_CNN1_CNN2_retrained`   | 0.0889  | 0.0861   | 0.0349      |
| from scratch + frame losses      | `from_scratch_frame_losses`   | 0.1041  | 0.1019   | 0.0544      |
