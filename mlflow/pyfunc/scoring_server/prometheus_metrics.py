"""
Opt-in Prometheus metrics for the scoring server.

Enabled by setting the ``prometheus_multiproc_dir`` environment variable (the same variable the
tracking server uses). Metrics are collected with ``prometheus_client`` in multiprocess mode, so a
single ``GET /metrics`` scrape aggregates all uvicorn workers.

Limitation: metric files of workers that have died are not cleaned up beyond what
``prometheus_client`` provides. The directory should be emptied before the server starts.
"""

import logging
import os
import time

_logger = logging.getLogger(__name__)

# Mirrors mlflow.server.constants.PROMETHEUS_EXPORTER_ENV_VAR; not imported because importing
# mlflow.server builds the tracking Flask app.
PROMETHEUS_MULTIPROC_DIR_ENV_VAR = "prometheus_multiproc_dir"
REQUEST_COUNTER_NAME = "mlflow_scoring_server_http_request_total"
REQUEST_DURATION_NAME = "mlflow_scoring_server_http_request_duration_seconds"
INVOCATIONS_PATH_LABEL = "/invocations"
OTHER_PATH_LABEL = "other"
_UNTRACKED_PATHS = frozenset({"/ping", "/health", "/version", "/metrics"})
_DURATION_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    float("inf"),
)
# Zero-valued children created at setup so both metrics appear in the first scrape.
_PREINITIALIZED_COUNTER_LABELS = (("/invocations", "200"), ("other", "404"))
_PREINITIALIZED_HISTOGRAM_LABELS = ("/invocations", "other")

# Lazily set to (Counter, Histogram).
_metrics = None


def setup_prometheus_metrics(app) -> bool:
    """
    Register the metrics middleware and the ``GET /metrics`` route on ``app`` if the
    ``prometheus_multiproc_dir`` environment variable is set.

    Returns:
        True if the metrics middleware and ``GET /metrics`` were registered on ``app``.
    """
    global _metrics

    multiproc_dir = os.environ.get(PROMETHEUS_MULTIPROC_DIR_ENV_VAR)
    if not multiproc_dir:
        return False

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
            "The '%s' environment variable is set, but the 'prometheus_client' package is not "
            "installed. Prometheus metrics are disabled. Install it with "
            "`pip install prometheus-client` to enable them.",
            PROMETHEUS_MULTIPROC_DIR_ENV_VAR,
        )
        return False

    from fastapi import Request
    from fastapi.responses import Response

    os.makedirs(multiproc_dir, exist_ok=True)

    if _metrics is None:
        _metrics = (
            Counter(
                REQUEST_COUNTER_NAME,
                "Total number of HTTP requests handled by the scoring server.",
                labelnames=("path", "status"),
                registry=None,
            ),
            Histogram(
                REQUEST_DURATION_NAME,
                "HTTP request latency of the scoring server in seconds.",
                labelnames=("path",),
                buckets=_DURATION_BUCKETS,
                registry=None,
            ),
        )
    counter, histogram = _metrics

    for path, status in _PREINITIALIZED_COUNTER_LABELS:
        counter.labels(path=path, status=status)
    for path in _PREINITIALIZED_HISTOGRAM_LABELS:
        histogram.labels(path=path)

    @app.middleware("http")
    async def prometheus_metrics_middleware(request: Request, call_next):
        if request.url.path in _UNTRACKED_PATHS:
            return await call_next(request)

        is_invocations = request.url.path == INVOCATIONS_PATH_LABEL
        path_label = INVOCATIONS_PATH_LABEL if is_invocations else OTHER_PATH_LABEL
        start = time.perf_counter()
        status_label = "500"
        try:
            response = await call_next(request)
            status_label = str(response.status_code)
            return response
        finally:
            counter.labels(path=path_label, status=status_label).inc()
            histogram.labels(path=path_label).observe(time.perf_counter() - start)

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return True
