# coding=utf-8
from logging.handlers import TimedRotatingFileHandler

import os
import sys
import logging


class LoggerFactory:

    @staticmethod
    def create_logger(name=None, level=logging.INFO):
        """create a logger

        Args:
            name (str): name of the logger
            level: level of logger

        Raises:
            ValueError is name is None
        """

        if name is None:
            raise ValueError("name for logger cannot be None")

        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] "
                                      "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s")

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger_.addHandler(ch)
        return logger_

    @staticmethod
    def create_logger_with_file(log_file_path: str = None, logger_level=logging.INFO):
        logger_inner = logging.getLogger()
        logger_inner.setLevel(logger_level)
        logger_inner.propagate = True

        formatter = logging.Formatter(fmt="[%(asctime)s] [%(filename)s:%(lineno)s - %(levelname)s] %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")

        # TimedRotatingFileHandler
        if log_file_path:
            basedir = os.path.dirname(log_file_path)
            if not os.path.isdir(basedir):
                os.makedirs(basedir, exist_ok=True)
            handler_file = TimedRotatingFileHandler(log_file_path, when="d", interval=1, backupCount=30)
            handler_file.setFormatter(formatter)
            logger_inner.addHandler(handler_file)

        # StreamHandler
        handler_console = logging.StreamHandler()
        handler_console.setFormatter(formatter)
        logger_inner.addHandler(handler_console)
        return logger_inner
