import os
import platform
import subprocess
from dataclasses import dataclass
from subprocess import Popen
from collections import namedtuple, Counter
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from itertools import product
from math import log10
from pathlib import Path
from time import sleep
from typing import Dict, List, Tuple
from datetime import timedelta

from lightning_trainable.launcher.utils import send_telegram_message
from tqdm import tqdm
from yaml import dump

ConfigWithCount = namedtuple("ConfigWithCount", ["config", "count"])


@dataclass
class RunResult:
    config: Dict
    return_code: int | str
    stdout: bytes
    stderr: bytes


class GridLauncher:
    def __init__(self, telegram_info: Dict = None):
        self.running_processes: List[Popen] = []
        self.telegram_info = telegram_info

    def send_message(self, message):
        if self.telegram_info is not None:
            send_telegram_message(message, **self.telegram_info)

    def run_configuration(self, config: List[Path | str | Tuple[str, object]], num_threads: int = None,
                          connect_debug: int = None, verbose=False, cli_args=None):
        """
        Runs a single configuration using lightning_trainable.launcher.fit
        in a subprocess and waits for the result.
        """
        arguments = []
        if connect_debug is not None:
            arguments.append("--pycharm-debug")
            arguments.append(str(connect_debug))
        if cli_args is not None:
            arguments.extend(cli_args)

        if num_threads is not None:
            config = config + [("num_threads", num_threads)]
        if len(config) > 0:
            for value in config:
                if isinstance(value, tuple):
                    key, value = value
                    if isinstance(value, type):
                        value = f"{value.__module__}.{value.__name__}"
                    arguments.append(f'{key}={dump(value)}')
                else:
                    arguments.append(str(value))

        out = None if verbose else subprocess.PIPE
        with Popen(['python', '-m', 'lightning_trainable.launcher.fit', *arguments],
                   stdout=out, stderr=out,
                   # Signals to controller are not passed to runner
                   preexec_fn=None if platform.system() == "Windows" else os.setpgrp) as process:
            self.running_processes.append(process)
            stdout, stderr = process.communicate()
            self.running_processes.remove(process)
            return RunResult(config=config, return_code=process.poll(), stdout=stdout, stderr=stderr)

    def grid_spec_to_list(self, config_spec: Dict[str, list] | List[list | Tuple[str, list]]):
        """
        Converts a grid of specifications to a list of configurations.

        Each specification can be a value or a list of values to be passed
        directly to the script or a tuple of a key and a value or a list of values.

        For example, the following code fits a BasicTrainable with the config file
        specified. The num_threads value is varied over 1, 2 and 4:
        grid_launcher = GridLauncher()
        grid_launcher.grid_spec_to_list([
            ("model", "tests.test_launcher.BasicTrainable"),
            "test_launcher_config.yaml",
            ("num_threads", [1, 2, 4]),
        ])
        """
        configs = []

        fake_keys = set()

        # Create fake keys for non-tuple entries
        dict_args = []
        if isinstance(config_spec, dict):
            config_spec = config_spec.items()
        for entry in config_spec:
            if isinstance(entry, tuple):
                dict_args.append(entry)
            else:
                fake_key = f"fake_key_{len(fake_keys)}"
                fake_keys.add(fake_key)
                dict_args.append((fake_key, entry))
        dict_args = dict(dict_args)

        # If a config entry is not a list of values, make it one
        def ensure_list(value):
            if isinstance(value, list):
                return value
            return [value]

        # Create all possible combinations, removing fake keys
        config_keys = list(dict_args.keys())
        for config_values in product(*map(ensure_list, dict_args.values())):
            config = [
                value if key in fake_keys else (key, value)
                for key, value in zip(config_keys, config_values)
            ]
            configs.append(config)
        return configs

    def start_runs(self, configs: List[List[Path | str]], num_parallel_runs=None,
                   num_threads=None, connect_debug: int = None, verbose=False,
                   cli_args=None, sleep_while_parallel: float = 0.0):
        """
        Starts a number of runs in parallel and returns the futures.
        """
        if num_parallel_runs is None:
            if num_threads is None:
                num_parallel_runs = 1
            else:
                num_parallel_runs = max(1, os.cpu_count() // num_threads - 1)

        pool = ThreadPoolExecutor(num_parallel_runs)
        futures = []
        for i, config in enumerate(configs):
            if i + 1 < num_parallel_runs:
                # Sleep while runs start immediately to prevent race conditions
                sleep(i * sleep_while_parallel)
            futures.append(pool.submit(
                self.run_configuration,
                config=config, num_threads=num_threads,
                connect_debug=connect_debug, verbose=verbose,
                cli_args=cli_args
            ))
        return pool, futures

    def run_configs_and_wait(self,
                             configs: List[List[Path | str]], num_parallel_runs=None,
                             num_threads=None, connect_debug: int = None, verbose=False,
                             cli_args=None, sleep_while_parallel: float = 0.0) -> List[RunResult]:
        """
        Runs a list of configurations in parallel and waits for the results.
        """
        pool, futures = self.start_runs(
            configs,
            num_parallel_runs=num_parallel_runs, num_threads=num_threads,
            connect_debug=connect_debug, verbose=verbose, cli_args=cli_args,
            sleep_while_parallel=sleep_while_parallel
        )
        interrupted_count = 0
        while True:
            try:
                results = self.fetch_results(futures)
                break
            except KeyboardInterrupt:
                interrupted_count += 1
                if interrupted_count == 1:
                    # Cancel future runs
                    pool.shutdown(wait=False, cancel_futures=True)
                    # Pool shutdown does not mark futures as_completed
                    # https://github.com/python/cpython/issues/87893
                    for f in tqdm(futures, desc="Cancelling future runs"):
                        if f.cancelled():
                            f.set_running_or_notify_cancel()
                    print("Stopped all pending experiments.")
                    print("Hit Ctrl-C again to cancel running experiments.")
                elif interrupted_count == 2:
                    # Cancel current runs
                    for process in tqdm(self.running_processes, desc="Killing processes"):
                        process.kill()
                    print("Stopped all running experiments.")
        if interrupted_count > 2:
            raise KeyboardInterrupt
        # Wait for remaining processes
        pool.shutdown(wait=True)

        # Print results
        status_counts = status_count_counter(results)
        print(f"Done running {len(configs)} experiments: {status_counts}")
        if len(set(status_counts) - {0}) > 0:
            print(f"Total: {sum(value for key, value in status_counts.items() if key != 0)} FAILED!")
        else:
            print("All succeeded :D")
        self.send_message(f"Launcher done: {format_status_counts(status_counts)}")
        return results

    def fetch_results(self, futures, timeout=None):
        """
        Fetches the results from a list of futures.
        """
        last_elapsed = 60
        results = []
        with tqdm(as_completed(futures, timeout=timeout), total=len(futures), smoothing=0) as pbar:
            for future in pbar:
                if future.cancelled():
                    result = RunResult(None, "cancelled", None, None)
                else:
                    result: RunResult = future.result()
                results.append(result)

                status_counts = status_count_counter(results)
                result_code = result.return_code

                # Status codes
                elapsed = pbar.format_dict["elapsed"]
                status_str = format_status_counts(status_counts)
                pbar.set_description(status_str)

                # Time as HH:MM:SS
                elapsed_delta = timedelta(seconds=elapsed)
                elapsed_delta_str = str(elapsed_delta).split(".")[0]
                if result_code not in [0, "cancelled"] and status_counts[result_code] == 10 ** int(
                        log10(status_counts[result_code])):
                    self.send_message(
                        f"Code {result_code}: {status_counts[result_code]} "
                        f"failed after {elapsed_delta_str}.\n"
                        f"Total: {status_str}"
                    )
                elif result_code == 0 and elapsed > min(2 * 60 * 60, last_elapsed * 2):
                    self.send_message(
                        f"{status_counts[result_code]} succeeded after {elapsed_delta_str}.\n"
                        f"Total: {status_str}"
                    )
                    last_elapsed = elapsed
        return results


def status_count_counter(results: List[RunResult]) -> Counter:
    return Counter(result.return_code for result in results)


def format_status_counts(status_counts: Counter) -> str:
    return ", ".join(f"{value}x {key}" for key, value in status_counts.items())
