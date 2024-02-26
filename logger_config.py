import logging
import os


# +
# Setup logger initially without file handler
def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:  # Prevent adding duplicate handlers if called multiple times
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    return logger


def add_file_handler(log_file):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(log_format)
    logger = logging.getLogger()  # Retrieve the already configured logger
    logger.addHandler(file_handler)


# -

logger = _setup_logger()
