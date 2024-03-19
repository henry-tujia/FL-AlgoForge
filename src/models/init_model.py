from omegaconf import DictConfig, OmegaConf
import inspect
import importlib


def flatten_dict(d):
    items = {}
    for k, v in d.items():
        if isinstance(v, dict):
            items.update(flatten_dict(v))
        else:
            items[k] = v
    return items


class Init_Model:
    def __init__(self, cfg: DictConfig) -> None:
        model_path = cfg.models._target_
        module_path = model_path.rsplit(".", 1)[0]
        module_name = model_path.rsplit(".", 1)[1]
        module = importlib.import_module(module_path)
        model_obj = getattr(module, module_name)
        needed_params = inspect.signature(model_obj).parameters

        cfg_dict = flatten_dict(OmegaConf.to_container(cfg, resolve=False))

        kwargs = {params: cfg_dict.get(params, None) for params in needed_params}

        self.model = model_obj(**kwargs)
