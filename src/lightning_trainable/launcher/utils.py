from pathlib import Path
from typing import List
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from yaml import safe_load


def parse_config_dict(config_spec: List[Path | str]):
    hparams = {}
    for arg in config_spec:
        if isinstance(arg, Path) or (
                any(
                    arg.endswith(suffix) for suffix in [".yaml", ".yml", ".json"]
                ) and "=" not in arg
        ):
            # Read multiple entries from .yaml file
            with open(arg, "r") as file:
                new_hparams = safe_load(file)
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
    return hparams


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
