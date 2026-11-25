"""One live, UTC-timestamped text log per training invocation."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from tempfile import NamedTemporaryFile
from time import gmtime
from typing import Literal

from src.config import LogSettings
from src.paths import ENV_FILE


@contextmanager
def run_log(kind: Literal["grid", "series"]) -> Iterator[logging.Logger]:
    directory = LogSettings(_env_file=ENV_FILE).log_dir
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"{kind}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-"
    with NamedTemporaryFile(dir=directory, prefix=prefix, suffix=".log", mode="w",
                            encoding="utf-8", delete=False) as stream:
        logger = logging.Logger(kind, level=logging.INFO)
        handler = logging.StreamHandler(stream)
        formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
        formatter.converter = gmtime
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        try:
            print(f"Run log: {stream.name}", flush=True)
            yield logger
        except KeyboardInterrupt:
            logger.warning("%s interrupted", kind.capitalize(), exc_info=True)
            raise
        except Exception:
            logger.exception("%s failed", kind.capitalize())
            raise
        finally:
            logger.removeHandler(handler)
            handler.close()
