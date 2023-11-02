from pathlib import Path
from typing import List, Any, Tuple, Dict
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from yaml import safe_load


def parse_config_dict(config_spec: Dict[str, Any] | List[Path | str | Tuple[str, Any]], hparams: dict = None):
    if hparams is None:
        hparams = {}
    if isinstance(config_spec, dict):
        config_spec = config_spec.items()
    for arg in config_spec:
        if isinstance(arg, tuple):
            key, value = arg
            new_hparams = {key: value}
        elif isinstance(arg, Path) or (
                any(
                    arg.endswith(suffix) for suffix in [".yaml", ".yml", ".json"]
                ) and ("/" in arg or "=" not in arg)  # Load from file if it contains "/" but not "="
        ):
            # Read multiple entries from .yaml file
            with open(arg, "r") as file:
                new_hparams = safe_load(file)
        else:
            # Read single entry from command line
            parts = arg.split("=", 1)
            if len(parts) != 2:
                raise ValueError(f"Config arg {arg!r}.")
            key, value = parts
            new_hparams = {key: safe_load(value)}

        # Merge in new parameters
        for key, value in new_hparams.items():
            hparam_level = hparams
            key_path = key.split(".")
            for i, key_entry in enumerate(key_path[:-1]):
                try:
                    hparam_level = dict_list_get(hparam_level, key_entry)
                except (KeyError, IndexError) as e:
                    raise e.__class__(f"Key path {'.'.join(key_path[:i + 1])!r} not found.") from e
            dict_list_set(hparam_level, key_path[-1], value)
    return hparams


def dict_list_get(dl: dict | list, item):
    if isinstance(dl, list):
        return dl[int(item)]
    return dl[item]


def dict_list_set(dl: dict | list, item, value):
    if isinstance(dl, list):
        if item == "+":
            dl.append(value)
        else:
            dl[int(item)] = value
    else:
        if item == "!":
            dl.update(value)
        else:
            dl[item] = value


def send_telegram_message(message: str, token: str, chats: List[int]):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        for chat_id in chats:
            params = {
                "chat_id": chat_id,
                "text": message
            }
            request = Request(url, urlencode(params).encode())
            urlopen(request).read().decode()
    except FileNotFoundError:
        pass
