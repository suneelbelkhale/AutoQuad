import sys
import os
import os.path
from colorlog import ColoredFormatter
import logging

import time

#writes all print statements to file
class Logger:
    GLOBAL_LOGGER_NAME = '_global_logger'

    _color_formatter = ColoredFormatter(
        "%(asctime)s %(levelname)s: %(message)s",
        datefmt='%m-%d %H:%M:%S',
        reset=False,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'white,bg_red',
        },
        secondary_log_colors={},
        style='%'
    )

    _normal_formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s',
        datefmt='%m-%d %H:%M:%S',
        style='%'
    )

    def __init__(self, log_file_prefix, display_name="logger_1", console_lvl='debug'):
        self.log_file_path = log_file_prefix + "_" + time.strftime('%H:%M_%m-%d-%Y', time.localtime()) + ".log"

        if os.path.isfile(self.log_file_path):
            os.rename(self.log_file_path, self.log_file_path + ".bak")

        # self.log_file = open(self.log_file_path, "w")
        self.file_handler = logging.FileHandler(self.log_file_path)
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(Logger._color_formatter)

        if console_lvl == 'debug':
            console_lvl = logging.DEBUG
        elif console_lvl == 'info':
            console_lvl = logging.INFO
        elif console_lvl == 'warn' or console_lvl == 'warning':
            console_lvl = logging.WARN
        elif console_lvl == 'error':
            console_lvl = logging.ERROR
        elif console_lvl == 'fatal' or console_lvl == 'critical':
            console_lvl = logging.CRITICAL
        else:
            raise ValueError('unknown logging level %s' % str(console_lvl))

#        self.console_handler = logging.StreamHandler()
#        self.console_handler.setLevel(console_lvl)
#        self.console_handler.setFormatter(Logger._color_formatter)

        self.logger = logging.getLogger(display_name)
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            #self.logger.addHandler(self.console_handler)
            self.logger.addHandler(self.file_handler)

    def print(self, *args, lvl='debug'):
        msg = ""
        for el in args:
            msg += str(el) + " "
        lvl = lvl.lower().strip()
        if lvl == 'debug':
            self.logger.debug(msg)
        elif lvl == 'info':
            self.logger.info(msg)
        elif lvl == 'warn' or lvl == 'warning':
            self.logger.warn(msg)
        elif lvl == 'error':
            self.logger.error(msg)
        elif lvl == 'fatal' or lvl == 'critical':
            self.logger.critical(msg)
        else:
            raise ValueError('unknown logging level %s' % str(lvl))

        sys.stdout.flush()




