# Copyright 2025 Matteo Bunino, CERN
#
# Original work Copyright 2021-2025 Joosep Pata, Eric Wulff, Farouk Mokhtar,
# Javier Duarte, Aadi Tepper, Ka Wa Ho, & Lars Sørlie
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# File adapted from particleflow (https://github.com/jpata/particleflow/tree/v2.2.0)
# for the itwinai plugin (https://github.com/matbun/mlpf-itwinai-plugin)

import logging
from functools import lru_cache


def _logging(rank, _logger, msg):
    """Will log the message only on rank 0 or cpu."""
    if (rank == 0) or (rank == "cpu"):
        _logger.info(msg)


def _configLogger(name, filename=None, loglevel=logging.INFO):
    # define a Handler which writes INFO messages or higher to the sys.stdout
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    if filename:
        logfile = logging.FileHandler(filename)
        logfile.setLevel(loglevel)
        logfile.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logger.addHandler(logfile)


class ColoredLogger:
    color_dict = {
        "black": "\033[0;30m",
        "red": "\033[0;31m",
        "green": "\033[0;32m",
        "orange": "\033[0;33m",
        "blue": "\033[0;34m",
        "purple": "\033[0;35m",
        "cyan": "\033[0;36m",
        "lightgray": "\033[0;37m",
        "darkgray": "\033[1;30m",
        "lightred": "\033[1;31m",
        "lightgreen": "\033[1;32m",
        "yellow": "\033[1;33m",
        "lightblue": "\033[1;34m",
        "lightpurple": "\033[1;35m",
        "lightcyan": "\033[1;36m",
        "white": "\033[1;37m",
        "bold": "\033[1m",
        "endcolor": "\033[0m",
    }

    def __init__(self, name):
        self.logger = logging.getLogger(name)

    def colorize(self, msg, color):
        return self.color_dict[color] + msg + self.color_dict["endcolor"]

    def debug(self, msg, *args, color=None, **kwargs):
        if color:
            msg = self.colorize(msg, color)
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, color=None, **kwargs):
        if color:
            msg = self.colorize(msg, color)
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, color=None, **kwargs):
        if color:
            msg = self.colorize(msg, color)
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, color=None, **kwargs):
        if color:
            msg = self.colorize(msg, color)
        self.logger.error(msg, *args, **kwargs)


_logger = ColoredLogger("mlpf")


@lru_cache(10)
def warn_once(msg, logger=_logger):
    # Keep track of 10 different messages and then warn again
    logger.warning(msg)
