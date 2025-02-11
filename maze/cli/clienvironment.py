import dataclasses
import enum
import logging

import click


@enum.unique
class LogLevel(enum.Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


LOG_LEVEL_MAP = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARNING: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
    LogLevel.FATAL: logging.FATAL,
}


@dataclasses.dataclass
class CliEnvironment:
    log_level: LogLevel = LogLevel.INFO
    logger: logging.Logger = logging.getLogger("maze_cli")


pass_env = click.make_pass_decorator(CliEnvironment, ensure=True)
