import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

import mlflow
from mlflow.pyfunc import PythonModel, scoring_server
from mlflow.pyfunc.scoring_server.prometheus_metrics import (
    REQUEST_COUNTER_NAME,
    REQUEST_LATENCY_NAME,
)
from mlflow.utils.os import is_windows

from tests.helper_functions import pyfunc_scoring_endpoint

_ENV_VAR_NAMES = ("prometheus_multiproc_dir", "PROMETHEUS_MULTIPROC_DIR")
_VALID_PAYLOAD = json.dumps({"instances": [[1, 2], [3, 4]]})


class _EchoModel(PythonModel):
    def predict(self, context, model_input, params=None):
        return model_input


@pytest.fixture(autouse=True)
def restore_environ():
    # `scoring_server.init` and the metrics setup mutate os.environ
    snapshot = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(snapshot)


@pytest.fixture
def model_path(tmp_path):
    path = str(tmp_path / "model")
    mlflow.pyfunc.save_model(path, python_model=_EchoModel())
    return path


@pytest.fixture
def multiproc_dir(tmp_path, monkeypatch):
    path = tmp_path / "prometheus"
    path.mkdir()
    for name in _ENV_VAR_NAMES:
        monkeypatch.setenv(name, str(path))
    return path


@pytest.fixture
def prometheus_enabled(multiproc_dir, monkeypatch):
    prometheus_client = pytest.importorskip("prometheus_client")
    # Multiprocess mode is chosen when `prometheus_client.values` is first imported, which may
    # have happened before the env var was set.
    monkeypatch.setattr(
        prometheus_client.values, "ValueClass", prometheus_client.values.MultiProcessValue()
    )
    return multiproc_dir


def _make_client(model_path):
    return TestClient(scoring_server.init(mlflow.pyfunc.load_model(model_path)))


def _post(client, data=_VALID_PAYLOAD, content_type="application/json"):
    return client.post("/invocations", content=data, headers={"Content-Type": content_type})


def _parse_samples(client):
    from prometheus_client.parser import text_string_to_metric_families

    response = client.get("/metrics")
    assert response.status_code == 200
    return [
        sample
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
    ]


def _counter_by_status(samples, path="/invocations"):
    return {
        s.labels["status"]: s.value
        for s in samples
        if s.name == REQUEST_COUNTER_NAME and s.labels["path"] == path
    }


def _histogram_count(samples, path="/invocations"):
    return sum(
        s.value
        for s in samples
        if s.name == f"{REQUEST_LATENCY_NAME}_count" and s.labels["path"] == path
    )


def test_metrics_disabled_by_default(model_path, monkeypatch):
    for name in _ENV_VAR_NAMES:
        monkeypatch.delenv(name, raising=False)

    app = scoring_server.init(mlflow.pyfunc.load_model(model_path))
    client = TestClient(app)

    assert len(app.user_middleware) == 1
    assert "/metrics" not in {getattr(route, "path", None) for route in app.routes}
    assert client.get("/metrics").status_code == 404
    assert _post(client).status_code == 200


def test_metrics_enabled_records_status_codes(model_path, prometheus_enabled):
    client = _make_client(model_path)

    assert _post(client).status_code == 200
    assert _post(client, data="{not json").status_code == 400
    assert _post(client, data="whatever", content_type="text/xml").status_code == 415

    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")

    samples = _parse_samples(client)
    assert _counter_by_status(samples) == {"200": 1, "400": 1, "415": 1}
    assert _histogram_count(samples) == 3


def test_metrics_enabled_registers_middleware_after_timeout(model_path, prometheus_enabled):
    app = scoring_server.init(mlflow.pyfunc.load_model(model_path))
    assert len(app.user_middleware) == 2
    assert "/metrics" in {getattr(route, "path", None) for route in app.routes}


def test_metrics_records_timeout_as_504(model_path, prometheus_enabled, monkeypatch):
    monkeypatch.setenv("MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT", "1")

    def slow_invocations(*args, **kwargs):
        time.sleep(2)
        return scoring_server.InvocationsResponse(
            response="[]", status=200, mimetype="application/json"
        )

    monkeypatch.setattr(scoring_server, "invocations", slow_invocations)
    client = _make_client(model_path)

    assert _post(client).status_code == 504

    samples = _parse_samples(client)
    assert _counter_by_status(samples) == {"504": 1}
    assert _histogram_count(samples) == 1


def test_metrics_labels_are_bounded(model_path, prometheus_enabled):
    client = _make_client(model_path)

    unknown_paths = ["/foo", "/invocations/extra", "/users/123", "/users/456"]
    for path in unknown_paths:
        assert client.get(path).status_code == 404
    for path in ["/ping", "/health", "/version", "/metrics"]:
        assert client.get(path).status_code == 200
    assert _post(client).status_code == 200

    samples = _parse_samples(client)
    counter_samples = [s for s in samples if s.name == REQUEST_COUNTER_NAME]
    histogram_samples = [s for s in samples if s.name.startswith(f"{REQUEST_LATENCY_NAME}_")]
    assert counter_samples
    assert histogram_samples

    assert {s.labels["path"] for s in counter_samples + histogram_samples} <= {
        "/invocations",
        "other",
    }
    for s in counter_samples:
        assert set(s.labels) == {"path", "status"}
    assert {label for s in histogram_samples for label in s.labels} == {"path", "le"}

    assert _counter_by_status(samples, "other") == {"404": len(unknown_paths)}
    assert _counter_by_status(samples) == {"200": 1}
    assert _histogram_count(samples, "other") == len(unknown_paths)
    assert _histogram_count(samples) == 1


def test_missing_prometheus_client_disables_metrics(model_path, multiproc_dir, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "prometheus_client", None)

    with caplog.at_level(logging.WARNING, logger="mlflow.pyfunc.scoring_server.prometheus_metrics"):
        app = scoring_server.init(mlflow.pyfunc.load_model(model_path))
    client = TestClient(app)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("prometheus_client" in r.getMessage() for r in warnings)
    assert len(app.user_middleware) == 1
    assert client.get("/metrics").status_code == 404
    assert _post(client).status_code == 200


def test_metrics_aggregate_across_workers(model_path, multiproc_dir, monkeypatch):
    prometheus_client = pytest.importorskip("prometheus_client")
    values = prometheus_client.values

    # Metric children are created lazily on the first `.labels(...)` call, so each simulated
    # worker's ValueClass must stay patched while that worker handles its requests.
    monkeypatch.setattr(values, "ValueClass", values.MultiProcessValue(lambda: "w1"))
    client1 = _make_client(model_path)
    for _ in range(3):
        assert _post(client1).status_code == 200

    monkeypatch.setattr(values, "ValueClass", values.MultiProcessValue(lambda: "w2"))
    client2 = _make_client(model_path)
    for _ in range(2):
        assert _post(client2).status_code == 200
    assert _post(client2, data="{not json").status_code == 400

    assert (multiproc_dir / "counter_w1.db").exists()
    assert (multiproc_dir / "counter_w2.db").exists()

    for client in (client1, client2):
        samples = _parse_samples(client)
        assert _counter_by_status(samples) == {"200": 5, "400": 1}
        assert _histogram_count(samples) == 6


@pytest.mark.skipif(is_windows(), reason="Multi-worker uvicorn serving is not tested on Windows")
def test_metrics_aggregate_across_uvicorn_workers(model_path, tmp_path, monkeypatch):
    pytest.importorskip("prometheus_client")
    multiproc_dir = tmp_path / "prometheus-e2e"
    multiproc_dir.mkdir()
    for name in _ENV_VAR_NAMES:
        monkeypatch.setenv(name, str(multiproc_dir))

    max_requests = 200
    with pyfunc_scoring_endpoint(
        model_path, extra_args=["--env-manager", "local", "--workers", "2"]
    ) as endpoint:
        url = f"http://localhost:{endpoint._port}"
        sent = 0
        while len(list(Path(multiproc_dir).glob("counter_*.db"))) < 2 and sent < max_requests:
            # A fresh connection per request gives each worker a chance to accept it
            response = requests.post(
                f"{url}/invocations",
                data=_VALID_PAYLOAD,
                headers={"Content-Type": "application/json", "Connection": "close"},
            )
            assert response.status_code == 200
            sent += 1
        assert len(list(Path(multiproc_dir).glob("counter_*.db"))) >= 2, (
            f"Only one worker handled requests after {sent} attempts"
        )

        from prometheus_client.parser import text_string_to_metric_families

        response = requests.get(f"{url}/metrics")
        assert response.status_code == 200
        totals = defaultdict(float)
        for family in text_string_to_metric_families(response.text):
            for s in family.samples:
                if s.name == REQUEST_COUNTER_NAME:
                    totals[s.labels["status"]] += s.value
        assert dict(totals) == {"200": sent}
