import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)
    elif not logger.level:
        logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_Formatter())
        logger.addHandler(handler)

    return logger


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        parts = [
            f"[{self.formatTime(record, '%Y-%m-%d %H:%M:%S')}]",
            f"{record.levelname:<7}",
            f"{record.name}",
            record.getMessage(),
        ]
        msg = "  ".join(parts)

        if hasattr(record, "extra_fields") and record.extra_fields:
            extras = ", ".join(
                f"{k}={v}" for k, v in record.extra_fields.items()
            )
            msg += f"  ({extras})"

        if record.exc_info and record.exc_info[0]:
            msg += "\n" + self.formatException(record.exc_info)
        return msg


def log_info(logger: logging.Logger, message: str, **extra):
    logger.info(message, extra={"extra_fields": extra} if extra else None)


def log_warning(logger: logging.Logger, message: str, **extra):
    logger.warning(message, extra={"extra_fields": extra} if extra else None)


def log_error(logger: logging.Logger, message: str, **extra):
    logger.error(message, extra={"extra_fields": extra} if extra else None)
