import yaml
from typing import Union, Dict
import os
from .constants import CONFIG_ROOT


def load_config(file_path: Union[os.PathLike, str] = None) -> dict:
    '''
    Parameters
    ----------
    file_path: string or path or none
        If None, then the default configuration is loaded.
    Returns
    -------
        Configuration as a nested dictionary.
    '''
    if not file_path:
        file_path = os.path.join(CONFIG_ROOT, "config_default.yaml")
    with open(file_path) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)
