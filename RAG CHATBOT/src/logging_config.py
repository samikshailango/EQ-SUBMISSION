import logging
import os
from logging.handlers import RotatingFileHandler

from config import LOG_DIR, LOG_FILE, LOG_LEVEL, LOG_MAX_BYTES, LOG_BACKUP_COUNT

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger once: a rotating file handler plus a
    console handler. Safe to call multiple times (e.g. once from app.py
    and once from a module it imports) -- only configures on the first call.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    resolved_level = getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
    )
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(resolved_level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a module-level logger, configuring logging on first use if no
    entrypoint has done so yet (so this is safe to call from anywhere,
    including scripts run directly)."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
