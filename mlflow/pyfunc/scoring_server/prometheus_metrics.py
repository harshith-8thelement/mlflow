"""
Opt-in Prometheus metrics for the scoring server.

Metrics are enabled by setting the same environment variable the tracking server uses
(``prometheus_multiproc_dir``) to a directory. When it is unset, this module does nothing: no
import of ``prometheus_client``, no middleware and no ``/metrics`` route.

Limitation: in multiprocess mode, ``prometheus_client`` stores one set of metric files per worker
process in that directory. Files of workers that have exited are not cleaned up here (beyond what
``prometheus_client`` provides), so their last recorded values keep being included in ``/metrics``.
Use a fresh, empty directory on each server start.
"""

import logging
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

_logger = logging.getLogger(__name__)

# Mirrors mlflow/server/constants.py. Not imported from there to keep the scoring server's mlflow
# dependencies to the minimum (it runs inside the model's environment).
PROMETHEUS_EXPORTER_ENV_VAR = "prometheus_multiproc_dir"

REQUEST_COUNTER_NAME = "mlflow_scoring_server_http_request_total"
REQUEST_LATENCY_NAME = "mlflow_scoring_server_http_request_duration_seconds"

# Only these paths are used as label values. Every other path is recorded under OTHER_PATH_LABEL
# so that arbitrary request paths cannot create unbounded label values.
RECORDED_PATHS = frozenset({"/invocations"})
EXCLUDED_PATHS = frozenset({"/ping", "/health", "/version", "/metrics"})
OTHER_PATH_LABEL = "other"


def maybe_enable_prometheus_metrics(app: "FastAPI") -> bool:
    """
    Register the metrics middleware and the ``/metrics`` route on ``app`` if the
    ``prometheus_multiproc_dir`` environment variable is set.

    Must be called right after the timeout middleware is registered so that the metrics
    middleware is the outermost layer and sees the final status code, including a timeout's 504.

    Returns:
        True if the middleware and the ``/metrics`` route were registered, False otherwise.
    """
    multiproc_dir = os.environ.get(PROMETHEUS_EXPORTER_ENV_VAR)
    if not multiproc_dir:
        return False

    # prometheus_client reads the uppercase variable at import time to select multiprocess mode.
    os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", multiproc_dir)
    os.makedirs(multiproc_dir, exist_ok=True)

    try:
        from prometheus_client import (
            CONTENT_TYPE_LATEST,
            CollectorRegistry,
            Counter,
            Histogram,
            generate_latest,
            multiprocess,
        )
    except ImportError:
        _logger.warning(
            "The environment variable '%s' is set but the 'prometheus_client' package is not "
            "installed. Prometheus metrics are disabled and '/metrics' is not available. "
            "Install it with `pip install prometheus_client` to enable them.",
            PROMETHEUS_EXPORTER_ENV_VAR,
        )
        return False

    from fastapi import Request
    from fastapi.responses import Response

    # registry=None avoids registering in the global registry, so calling init() more than once
    # in a process does not raise a duplicate-timeseries error. Values are written to the
    # multiprocess directory and read back by MultiProcessCollector in /metrics.
    request_counter = Counter(
        REQUEST_COUNTER_NAME,
        "Total number of HTTP requests handled by the scoring server",
        labelnames=["path", "status"],
        registry=None,
    )
    request_latency = Histogram(
        REQUEST_LATENCY_NAME,
        "HTTP request latency of the scoring server in seconds",
        labelnames=["path"],
        registry=None,
    )

    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next):
        path = request.url.path
        if path in EXCLUDED_PATHS:
            return await call_next(request)

        label = path if path in RECORDED_PATHS else OTHER_PATH_LABEL
        start = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            request_counter.labels(path=label, status=status).inc()
            request_latency.labels(path=label).observe(time.perf_counter() - start)

    @app.get("/metrics")
    def metrics():
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry, path=multiproc_dir)
        return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return True
