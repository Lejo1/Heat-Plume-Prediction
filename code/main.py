import optuna
from optuna.trial import TrialState
import argparse
from pathlib import Path
from processing.training import training, run
from processing.training_e2e import training_e2e
from utils.utils_args import read_cla

PATH_DATA_PREP = Path("../../datasets_prep") # TODO: change to your path
PATH_MODELS_DIR = Path("../runs") # TODO: change to your path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=str, default=None)
    parser.add_argument("--hsearch", type=bool, default=False)
    args = parser.parse_args()
    args = vars(args)
    hsearch = args["hsearch"]

    args = read_cla(PATH_MODELS_DIR / args["destination"])
    
    if not hsearch:
        if args.get("pipeline") == "e2e":
            model = training_e2e(args, PATH_DATA_PREP)
        else:
            model = training(args, PATH_DATA_PREP)

    else:
        print("Study name: ", args["destination"])
        study = optuna.create_study(direction="minimize", storage=f"sqlite:///{PATH_MODELS_DIR}/TEST_STUDY.db", study_name="NAME", load_if_exists=True)
        study.optimize(lambda trial: run(trial, args, PATH_DATA_PREP), n_trials=1)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    print("Done")