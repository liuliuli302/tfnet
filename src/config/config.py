import os
from typing import Any, Dict
import yaml


class BasicConfig:
    """
    基本配置类，提供从yaml加载配置的方法
    """

    def __init__(self, **kwargs):
        # 动态处理所有传入的关键字参数
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**config_dict)

    @classmethod
    def load_config_from_file(cls, config_file_path: str):
        with open(config_file_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
