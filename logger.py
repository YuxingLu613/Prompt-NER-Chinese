import logging
import os
from config import Config

config=Config()

if not os.path.exists(config.log_path):
    os.makedirs(config.log_path)

logger=logging.getLogger("client_log")

logger.setLevel(logging.INFO)

stream_handler=logging.StreamHandler()
log_file_handler=logging.FileHandler(filename=os.path.join(config.log_path,config.log_name),encoding="utf-8")

formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(message)s")

stream_handler.setFormatter(formatter)
log_file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(log_file_handler)