import yaml
import requests
from opencompass.utils import get_logger
from copy import deepcopy

logger = get_logger(log_level='INFO')

class Config:

    def __init__(self, config: dict) -> None:
        self.__task_config = config
        self.__check_version()
    
    def get_config_dict(self):
        return deepcopy(self.__task_config)
    
    @staticmethod
    def from_file(file_path: str) -> 'Config':
        logger.info(f"loading config from {file_path}")
        with open(file_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            return Config(config)

    @staticmethod
    def from_url(url: str) -> 'Config':
        logger.info(f"loading config from {url}")
        if not url.startswith("http"):
            raise ValueError(f"invalid url: {url}")
        resp = requests.get(url)
        config = resp.content.decode("utf-8")
        logger.debug(config)
        config = yaml.load(config, Loader=yaml.FullLoader)
        return Config(config)

    def __check_version(self):
        if self.version != '1.0.0':
            raise ValueError(f"invalid config version: {self.version}")

    def dump_to(self, path: str) -> None:
        logger.info(f"dumping config to {path}")
        with open(path, 'w') as file:
            yaml.dump(self.__task_config, file, sort_keys=False,
                      allow_unicode=True, encoding="utf-8")

    @property
    def version(self):
        return self.__task_config["version"]

    @property
    def model(self):
        return self.__task_config["model"]

    @property
    def datasets(self):
        return self.__task_config["datasets"]

    @property
    def reporter(self):
        return self.__task_config["reporter"] or {}

    @property
    def cuda_device(self):
        device = self.__task_config["cuda_device"]
        if device is None:
            return "all"
        return str(device)

    @property
    def debug(self):
        return self.__task_config.get("debug") or False
    
    @property
    def generate_mode(self):
        return self.__task_config.get("generate_mode") or "model_pool"