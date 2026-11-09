"""
Opt-in Prometheus metrics for the scoring server.

Enabled by setting the ``prometheus_multiproc_dir`` environment variable (the same variable the
tracking server uses). When it is unset or empty, nothing in this module registers anything.

Limitation: metric files of dead workers are not cleaned up beyond what ``prometheus_client``
provides. Stale files in the multiprocess directory keep contributing to ``/metrics`` until the
directory is cleared (e.g. on server restart).
"""

import logging
import os
import time

# NB: Only stdlib imports at the top level. The scoring server runs inside the model's environment
# where prometheus_client may not be installed, so it is imported lazily and only when enabled.

_logger = logging.getLogger(__name__)

PROMETHEUS_MULTIPROC_DIR_ENV_VAR = "prometheus_multiproc_dir"  # same as mlflow.server.constants
REQUEST_COUNTER_NAME = "mlflow_scoring_server_http_request_total"
REQUEST_LATENCY_NAME = "mlflow_scoring_server_http_request_duration_seconds"
METRICS_PATH = "/metrics"
KNOWN_PATHS = frozenset({"/invocations", "/ping", "/health", "/version"})
OTHER_PATH_LABEL = "other"


def maybe_enable_prometheus_metrics(app) -> bool:
    """
    Register a metrics middleware and a ``GET /metrics`` route on ``app`` if the
    ``prometheus_multiproc_dir`` environment variable is set.

    Returns:
        True if the middleware and route were registered, False otherwise (the environment
        variable is unset or empty, or ``prometheus_client`` is not importable).
    """
    multiproc_dir = os.environ.get(PROMETHEUS_MULTIPROC_DIR_ENV_VAR)
    if not multiproc_dir:
        return False

    # prometheus_client picks its value class (multiprocess or not) at import time from the
    # upper-case variable, so it must be set before the first import.
    os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", multiproc_dir)

    try:
        import prometheus_client  # noqa: F401
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
            "The '%s' environment variable is set but 'prometheus_client' is not installed. "
            "Prometheus metrics are disabled. Install it with `pip install prometheus_client`.",
            PROMETHEUS_MULTIPROC_DIR_ENV_VAR,
        )
        return False

    from fastapi import Request
    from fastapi.responses import Response

    os.makedirs(os.environ["PROMETHEUS_MULTIPROC_DIR"], exist_ok=True)

    request_counter = Counter(
        REQUEST_COUNTER_NAME,
        "Total number of HTTP requests handled by the scoring server",
        ["path", "status"],
        registry=None,
    )
    request_latency = Histogram(
        REQUEST_LATENCY_NAME,
        "HTTP request latency of the scoring server in seconds",
        ["path"],
        registry=None,
    )

    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next):
        path = request.url.path
        if path == METRICS_PATH:
            return await call_next(request)

        label = path if path in KNOWN_PATHS else OTHER_PATH_LABEL
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            request_latency.labels(path=label).observe(time.perf_counter() - start)
            request_counter.labels(path=label, status=str(int(status))).inc()

    @app.get(METRICS_PATH)
    async def metrics():
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return True
