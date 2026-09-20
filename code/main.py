import optuna
from optuna.trial import TrialState
import argparse
from pathlib import Path
from processing.training import training, run
from processing.training_e2e import training_e2e, run_e2e
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
        # one study per search folder: its own sqlite file and study name, so searches do not mix and
        # an interrupted search continues where it stopped (load_if_exists).
        base = args["destination"]
        n_trials = int(args.get("n_trials", 50))
        pruner = optuna.pruners.MedianPruner(n_startup_trials=int(args.get("pruner_startup_trials", 3)),
                                             n_warmup_steps=int(args.get("pruner_warmup_epochs", 10)))
        objective = run_e2e if args.get("pipeline") == "e2e" else run
        print(f"Study name: {base.name} ({objective.__name__}), {n_trials} trials, storage {base}/hsearch.db")
        study = optuna.create_study(direction="minimize", storage=f"sqlite:///{base}/hsearch.db",
                                    study_name=base.name, load_if_exists=True, pruner=pruner)
        study.optimize(lambda trial: objective(trial, args, PATH_DATA_PREP), n_trials=n_trials)

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