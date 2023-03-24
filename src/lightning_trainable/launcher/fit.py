from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path

from yaml import full_load, safe_load

import torch

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--pycharm-debug", default=False, type=int,
                        help="Port of PyCharm remote debugger to connect to.")
    log_dir_group = parser.add_mutually_exclusive_group()
    log_dir_group.add_argument("--name", type=str,
                               help="Name of experiment. Experiment data will be stored in "
                                    "lightning_logs/`name`/version_X")
    log_dir_group.add_argument("--log-dir", type=str,
                               help="Experiment data will be stored `log_dir`'")
    parser.add_argument("config_args", type=str, nargs="*",
                        help="")
    args = parser.parse_args()

    if args.pycharm_debug:
        import pydevd_pycharm

        pydevd_pycharm.settrace('localhost', port=args.pycharm_debug, stdoutToServer=True, stderrToServer=True)

    hparams = {}
    for arg in args.config_args:
        if "=" not in arg and any(
                arg.endswith(suffix) for suffix in [".yaml", ".yml", ".json"]
        ):
            # Read multiple entries from .yaml file
            with open(arg, "r") as file:
                new_hparams = full_load(file)
        else:
            # Read single entry from command line
            key, value = arg.split("=")
            new_hparams = {key: safe_load(value)}

        # Merge in new parameters
        for key, value in new_hparams.items():
            hparam_level = hparams
            key_path = key.split(".")
            for key_entry in key_path[:-1]:
                hparam_level = hparam_level[key_entry]
            hparam_level[key_path[-1]] = value

    # Set number of threads (potentially move into trainable, but it's a global property)
    num_threads = hparams.get("num_threads", None)
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    # Load the model
    module_name, model_name = hparams.pop("model")
    module = import_module(module_name)
    model = getattr(module, model_name)(hparams=hparams)

    # Log path
    if args.name is not None:
        logger_kwargs = dict(
            save_dir="lightning_logs",
            name=args.name
        )
    elif args.save_dir is not None:
        save_path = Path(args.save_dir)
        logger_kwargs = dict(
            version=save_path.name,
            experiment_name=save_path.parent.name,
            save_dir=save_path.parent.parent
        )
    else:
        logger_kwargs = dict()

    # Fit the model
    model.fit(logger_kwargs=logger_kwargs)
