import os
import logging
import random
from typing import Sequence
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import pathlib
import re
import pandas
import yaml
import rich
import rich.syntax
import rich.tree
from rich.logging import RichHandler


def set_logger(log_path, loggername, mode="server"):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    if not isinstance(log_path, pathlib.Path):
        log_path = pathlib.Path(log_path)
    pathlib.Path.mkdir(log_path, exist_ok=True, parents=True)

    logger = logging.getLogger(loggername)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    log_path = log_path / (loggername + ".log")
    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        if mode == "scheduler":
            logger.addHandler(RichHandler())
    return logger


def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # NOTE: If you want every run to be exactly the same each time
    # uncomment the following lines
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def parser_log(path):
    pattern = r"[0-9]{2,}\.[0-9]*"

    content = path.read_text()

    matches = re.findall(pattern, content, re.MULTILINE)

    accs = [matches[i] for i in range(0, len(matches), 2)]

    df = pandas.DataFrame(data=accs, columns=["ACC"])
    df["Round"] = df.index

    df.to_csv(path.parent / (path.stem + ".csv"), sep="\t")


def find_log(conditions, path):
    res = []
    for log_file in path.rglob("server.log"):
        method_path = log_file.parent
        if conditions["method"] + ".yaml" in [
            x.stem + x.suffix for x in method_path.glob("*.yaml")
        ]:
            with open(method_path / "base.yaml", "r", encoding="utf-8") as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
                if all(
                    [
                        conditions["config"][key] == data[key]
                        for key in conditions["config"]
                    ]
                ):
                    res.append(method_path)
    return res


def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "federated_settings",
        "local_setting",
        "method",
        "datasets",
        "paths",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
    loggger: logging.Logger = None,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    :param cfg: A DictConfig composed by Hydra.
    :param print_order: Determines in what order config components are printed. Default is ``("data", "model",
    "callbacks", "logger", "trainer", "paths", "extras")``.
    :param resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
    :param save_to_file: Whether to export config to the hydra output folder. Default is ``False``.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else loggger.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)
    # cfg.paths.output_dir: pathlib.Path
    # save config tree to file
    if save_to_file:
        with (pathlib.Path(cfg.paths.output_dir) / "config_tree.log").open("w") as f:
            # with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=f)


if __name__ == "__main__":
    # parser_log(
    #     pathlib.Path(
    #         "/cpfs/user/haotan2/FL/Mutil-FL-Training-main/save/2023-09-27T01:25:05.430568/server.log"
    #     )
    # )
    # print(list(pathlib.Path("/cpfs/user/haotan2/FL/Mutil-FL-Training-main/save").glob("*")))
    print(
        find_log(
            conditions={
                "method": "fedavg",
                "config": {
                    "client_number": 100,
                    "client_sample": 0.1,
                    "partition_alpha": 0.1,
                },
            },
            path=pathlib.Path("/cpfs/user/haotan2/FL/Mutil-FL-Training-main/save"),
        )
    )
