
from pathlib import Path
import re


def find_version(root: str | Path = "lightning_logs", version: int = "last") -> int:
    root = Path(root)

    # Determine latest version number if "last" is passed as version number
    if version == "last":
        version_folders = [f for f in root.iterdir() if f.is_dir() and re.match(r"^version_(\d+)$", f.name)]

        if not version_folders:
            raise FileNotFoundError(f"Found no folders in '{root}' matching the name pattern 'version_<number>'")

        version_numbers = [int(re.match(r"^version_(\d+)$", f.name).group(1)) for f in version_folders]
        version = max(version_numbers)

    return version


def find_epoch_step(root: str | Path, epoch: int = "last", step: int = "last") -> (int, int):
    """
    Find epoch and step number for given checkpoint root. Checks if such a checkpoint exists.
    Note that this method *ignores* last.ckpt files (these are handled in find_checkpoint)

    @param root: Checkpoint root directory. Usually lightning_logs/version_i/checkpoints/
    @param epoch: Epoch number or "last"
    @param step: Step number or "last"
    @return: epoch and step numbers
    """
    root = Path(root)

    # get checkpoint filenames
    checkpoints = [f.name for f in root.glob("*.ckpt")]

    pattern = re.compile(r"^epoch=(\d+)-step=(\d+)\.ckpt$")

    # remove invalid files
    checkpoints = [cp for cp in checkpoints if pattern.match(cp)]

    if not checkpoints:
        raise FileNotFoundError(f"Found no checkpoints in '{root}' matching "
                                f"the name pattern 'epoch=<number>-step=<number>.ckpt'")

    # get epochs and steps as list
    matches = [pattern.match(cp) for cp in checkpoints]
    epochs, steps = zip(*[(int(match.group(1)), int(match.group(2))) for match in matches])

    if epoch == "last":
        epoch = max(epochs)
    elif epoch not in epochs:
        closest_epoch = min(epochs, key=lambda e: abs(e - epoch))
        raise FileNotFoundError(f"No checkpoint in '{root}' for epoch '{epoch}'. Closest is '{closest_epoch}'.")

    # keep only steps for this epoch
    steps = [s for i, s in enumerate(steps) if epochs[i] == epoch]

    if step == "last":
        step = max(steps)
    elif step not in steps:
        closest_step = min(steps, key=lambda s: abs(s - step))
        raise FileNotFoundError(f"No checkpoint in '{root}' for epoch '{epoch}', step '{step}'. "
                                f"Closest step for this epoch is '{closest_step}'.")

    return epoch, step


def find_checkpoint(root: str | Path = "lightning_logs", version: int = "last", epoch: int = "last", step: int = "last") -> str:
    """
    Helper method to find a lightning checkpoint based on version, epoch and step numbers.

    @param root: logs root directory. Usually "lightning_logs/" or "lightning_logs/<experiment_name>"
    @param version: version number or "last"
    @param epoch: epoch number or "last"
    @param step: step number or "last"
    @return: path to the checkpoint, relative to root
    """
    root = Path(root)

    if not root.is_dir():
        raise ValueError(f"Checkpoint root directory '{root}' does not exist")

    # get existing version number or error
    version = find_version(root, version)

    checkpoint_folder = root / f"version_{version}" / "checkpoints"

    contents = "\n".join([str(p) for p in checkpoint_folder.iterdir()])

    if epoch == "last" and step == "last":
        # return last.ckpt if it exists
        checkpoint = checkpoint_folder / "last.ckpt"
        if checkpoint.is_file():
            return str(checkpoint)
        else:
            # TODO: remove debug error
            msg = f"""
            Could not find 'last.ckpt' in '{checkpoint_folder}'.
            Checkpoint folder exists? {checkpoint_folder.is_dir()}
            Checkpoint exists? {checkpoint.exists()}
            Checkpoint is file? {checkpoint.is_file()}
            Checkpoint path: {checkpoint}
            Checkpoint folder contents:
            {contents}
            """
            raise RuntimeError(msg)

    # get existing epoch and step number or error
    epoch, step = find_epoch_step(checkpoint_folder, epoch, step)

    checkpoint = checkpoint_folder / f"epoch={epoch}-step={step}.ckpt"

    return str(checkpoint)
