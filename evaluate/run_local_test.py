from opencompass.cli.main import main
from opencompass.Config import Config
import os 
import sys
from opencompass.utils import get_logger
sys.path.append(os.getcwd())

logger = get_logger(log_level='INFO')
local_test_config_file = './local_task.yaml'

def load_config(config_path=local_test_config_file):
    cfg = Config.from_file(config_path)
    return cfg

if __name__ == '__main__':
    cfg = load_config()
    cfg = cfg.get_config_dict()
    
    if 'local_model_path' not in cfg.keys():
        logger.error(f"You don't define local model path.")
        sys.exit(1)
    logger.info(f"Input Config is {cfg}")
    
    report_path = main(cfg, max_num_workers=32)
    logger.info("Suceessfully finish evaluation")

    
