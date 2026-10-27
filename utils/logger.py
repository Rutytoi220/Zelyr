"""Logging utilities for Zelyr."""

import logging
import os
from logging.handlers import RotatingFileHandler
from .config import get_config


def get_logger(name: str = "zelyr") -> logging.Logger:
    cfg = get_config()
    log_cfg = cfg.get("logging", {}) if isinstance(cfg, dict) else {}
    level_name = (log_cfg.get("level") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = log_cfg.get("file") or "zelyr.log"

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    try:
        fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=2)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        logger.warning("Could not create file handler for logging")

    return logger
