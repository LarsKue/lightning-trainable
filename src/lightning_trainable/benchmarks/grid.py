
from lightning_trainable import Trainable
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm


def benchmark_grid(models: list[type(Trainable)], parameters: list[dict], datasets: list[Dataset], **fit_kwargs)\
        -> pd.DataFrame:
    """
    Run benchmarks for multiple model types, or multiple parameter sets, across different datasets
    :param models: list of model classes to benchmark
    :param parameters: list of parameter sets to benchmark
    :param datasets: list of datasets to fit on
    :param eval_metrics: list of
    :return:
    """
    df = pd.DataFrame(columns=["Model", "Dataset", "Parameter Set"])

    for model_cls in tqdm(models):
        for params_idx, hparams in enumerate(tqdm(parameters)):
            for ds in tqdm(datasets):
                model = model_cls(hparams, train_data=ds, val_data=ds)
                metrics = model.fit(**fit_kwargs)
                row = {
                    "Model": model_cls.__name__,
                    "Dataset": ds.__class__.__name__,
                    "Parameter Set": params_idx,
                    **metrics,
                }
                df = df.append(row, ignore_index=True)

    return df
