from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path

from lightning_trainable.launcher.utils import parse_config_dict

import torch


def main(args=None):
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
    args = parser.parse_args(args)

    if args.pycharm_debug:
        import pydevd_pycharm

        pydevd_pycharm.settrace('localhost', port=args.pycharm_debug, stdoutToServer=True, stderrToServer=True)

    hparams = parse_config_dict(args.config_args)

    # Set number of threads (potentially move into trainable, but it's a global property)
    num_threads = hparams.pop("num_threads", None)
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    # Load the model
    module_name, model_name = hparams["model"].rsplit(".", 1)

    # Log path
    if args.name is not None:
        logger_kwargs = dict(
            save_dir="lightning_logs",
            name=args.name.format(
                model_name=model_name,
                **hparams
            )
        )
    elif args.log_dir is not None:
        save_path = Path(args.save_dir)
        logger_kwargs = dict(
            version=save_path.name,
            experiment_name=save_path.parent.name,
            save_dir=save_path.parent.parent
        )
    else:
        logger_kwargs = dict()

    # No "model" hparam
    del hparams["model"]
    module = import_module(module_name)
    model = getattr(module, model_name)(hparams=hparams)

    # Fit the model
    return model.fit(logger_kwargs=logger_kwargs)


if __name__ == '__main__':
    main()
