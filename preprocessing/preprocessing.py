from preprocessing.prepare_dataset import prepare_dataset, is_unprepared
import utils.utils_args as ua


def preprocessing(args:dict):
    print("Preparing dataset")
    # handling of case=="test"? TODO
    if is_unprepared(args["data_prep"]):
        if args["problem"] == "2stages":
            exit("2stages not implemented yet, use other branch")

        additional_inputs_unnormed = None

        info = ua.load_yaml(args["model"]/"info.yaml") if args["case"] != "train" else None
        info = prepare_dataset(args, info=info, additional_inputs=additional_inputs_unnormed)
            # handling of case=="test"? TODO

    else:
        info = ua.load_yaml(args["data_prep"]/"info.yaml") 
    print(f"Dataset prepared: {args['data_prep']}")

    if args["case"] == "train": # TODO also finetune?
        ua.save_yaml(info, args["destination"]/"info.yaml")