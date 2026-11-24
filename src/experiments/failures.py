"""Bounded diagnostic codes for the worker result and monitoring event."""

import errno

import torch
from mlflow.exceptions import MlflowException
from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout

from monitoring.events import Reason


class WorkerFailed(RuntimeError):
    def __init__(self, message: str, reason: Reason, run_id: str | None):
        super().__init__(message)
        self.reason = reason
        self.run_id = run_id


def failure_reason(error: BaseException) -> Reason:
    seen = set()
    tracking_request = False
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, WorkerFailed):
            return error.reason
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return "cuda_oom"
        if isinstance(error, OSError):
            if error.errno in (errno.EDQUOT, errno.ENOSPC):
                return "disk_full"
            if error.errno == errno.EROFS:
                return "write_error"
        if isinstance(error, MlflowException):
            tracking_request = True
            if error.error_code in ("TEMPORARILY_UNAVAILABLE", "ENDPOINT_UNAVAILABLE", "DEADLINE_EXCEEDED"):
                return "mlflow_unavailable"
        if tracking_request and isinstance(error, (RequestsConnectionError, Timeout)):
            return "mlflow_unavailable"
        error = error.__cause__ or error.__context__
    return "series_error"
