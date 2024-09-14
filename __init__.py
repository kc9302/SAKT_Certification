import torch
import configparser
import logging

from common.file_setting import find_datasets_path, get_pkl, make_check_points
from common.utils import match_sequence_length, set_optimizer, collate_fn, logging_sakt_config

# Operation flow sequence 1.
try:
    # setting device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # read config.ini
    config = configparser.ConfigParser()
    config.read("./config.ini")
except Exception as err:
    logging.error(err)
