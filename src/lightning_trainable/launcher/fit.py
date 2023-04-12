import re
from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path

from pytorch_lightning import LightningModule

from lightning_trainable.launcher.utils import parse_config_dict

import torch


def main(args=None):
    parser = ArgumentParser()
    parser.add_argument("--pycharm-debug", default=False, type=int,
                        help="Port of PyCharm remote debugger to connect to.")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--start-from", type=Path,
                              help="Load weights from a specified checkpoint. "
                                   "You must specify at least a `model` config argument. "
                                   "All other config arguments overwrite the values in the stored checkpoint.")
    resume_group.add_argument("--continue-from", type=Path,
                              help="Load the state of model, trainer, optimizer, ... from the given checkpoint "
                                   "and resume training. "
                                   "You must specify at least a `model` config argument. "
                                   "All other config arguments overwrite the values in the stored checkpoint."
                              )
    parser.add_argument("--loose-load-state-dict", action="store_true", default=False,
                        help="When loading a state dict, set `strict`=False")
    parser.add_argument("--gradient-regex", type=str, default=None,
                        help="Parameter names must contain regex expression to have gradient applied.")
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

    # Merge hparams from checkpoint and configs
    checkpoint = None
    fit_kwargs = {}
    if args.start_from is not None:
        checkpoint = torch.load(args.start_from)
    elif args.continue_from is not None:
        checkpoint = torch.load(args.continue_from)
        fit_kwargs["ckpt_path"] = args.continue_from
    if checkpoint is None:
        hparams = {}
    else:
        hparams = checkpoint[LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]
    hparams = parse_config_dict(args.config_args, hparams)

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
    module = import_module(module_name)
    del hparams["model"]
    model_class = getattr(module, model_name)
    model = model_class(hparams=hparams)
    from lightning_trainable import Trainable
    assert isinstance(model, Trainable)

    # Load weights from checkpoint
    if checkpoint is not None:
        strict = not args.loose_load_state_dict
        keys = model.load_state_dict(checkpoint["state_dict"], strict=strict)
        if not strict:
            missing_keys, unexpected_keys = keys
            if len(missing_keys) > 0:
                print(f"When loading, the following keys were not found: {missing_keys}")
            if len(unexpected_keys) > 0:
                print(f"When loading, the following keys were not used: {unexpected_keys}")

    # Apply gradient regex
    if args.gradient_regex is not None:
        deactivated_parameters = remaining_parameters = 0
        for name, parameter in model.named_parameters():
            if not re.search(args.gradient_regex, name):
                parameter.requires_grad = False
                deactivated_parameters += 1
            else:
                remaining_parameters += 1
        print(f"Deactivated {deactivated_parameters} parameters, {remaining_parameters} parameters left as is.")

    # Fit the model
    return model.fit(logger_kwargs=logger_kwargs, fit_kwargs=fit_kwargs)


if __name__ == '__main__':
    main()
