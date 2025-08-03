# src/compositeimage/config.py
from omegaconf import OmegaConf
from functools import lru_cache

@lru_cache(maxsize=1)
def load_config(path: str = "params.yaml"):
    return OmegaConf.load(path)