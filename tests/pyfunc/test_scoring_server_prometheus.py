import json
import logging
import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

import mlflow.pyfunc
import mlflow.sklearn
from mlflow.pyfunc import scoring_server
from mlflow.pyfunc.scoring_server._prometheus import (
    OTHER_PATH_LABEL,
    REQUEST_COUNTER_NAME,
    REQUEST_LATENCY_NAME,
)

GOOD_PAYLOAD = json.dumps({"inputs": [[1.0, 2.0], [3.0, 4.0]]})
JSON_HEADERS = {"Content-Type": scoring_server.CONTENT_TYPE_JSON}


class SleepModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        time.sleep(2)
        return model_input


@pytest.fixture(autouse=True)
def _restore_environ():
    # `init()` and the metrics setup write to `os.environ` directly
    snapshot = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(snapshot)


@pytest.fixture(scope="module")
def sklearn_model_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("model") / "model")
    model = LogisticRegression().fit(np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]]), [0, 0, 1])
    mlflow.sklearn.save_model(model, path)
    return path


@pytest.fixture
def sleep_model_path(tmp_path):
    path = str(tmp_path / "sleep_model")
    mlflow.pyfunc.save_model(path, python_model=SleepModel())
    return path


@pytest.fixture
def prometheus_dir(monkeypatch, tmp_path):
    prometheus_client = pytest.importorskip("prometheus_client")
    path = tmp_path / "prometheus"
    path.mkdir()
    # Force `monkeypatch` to record (and later restore) the upper-case variable that the server
    # sets itself for `prometheus_client`.
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", "placeholder")
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR")
    monkeypatch.setenv("prometheus_multiproc_dir", str(path))
    # `prometheus_client` selects its value class at import time; make sure values go to `path`
    # even if it was imported earlier, and start with a fresh per-test file cache.
    monkeypatch.setattr(
        prometheus_client.values, "ValueClass", prometheus_client.values.MultiProcessValue()
    )
    return path


def _make_client(model_path):
    return TestClient(scoring_server.init(mlflow.pyfunc.load_model(model_path)))


def _samples(client, name):
    from prometheus_client.parser import text_string_to_metric_families

    response = client.get("/metrics")
    assert response.status_code == 200
    return [
        s
        for family in text_string_to_metric_families(response.text)
        for s in family.samples
        if s.name == name
    ]


def _counter_values(client):
    return {
        (s.labels["path"], s.labels["status"]): s.value
        for s in _samples(client, REQUEST_COUNTER_NAME)
    }


def _histogram_counts(client):
    return {s.labels["path"]: s.value for s in _samples(client, REQUEST_LATENCY_NAME + "_count")}


def test_metrics_disabled_by_default(monkeypatch, sklearn_model_path):
    monkeypatch.delenv("prometheus_multiproc_dir", raising=False)
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    client = _make_client(sklearn_model_path)

    assert len(client.app.user_middleware) == 1
    assert (
        client.post("/invocations", content=GOOD_PAYLOAD, headers=JSON_HEADERS).status_code == 200
    )
    assert client.get("/metrics").status_code == 404
    assert "PROMETHEUS_MULTIPROC_DIR" not in os.environ


def test_metrics_disabled_when_env_var_is_empty(monkeypatch, sklearn_model_path):
    monkeypatch.setenv("prometheus_multiproc_dir", "")
    client = _make_client(sklearn_model_path)

    assert len(client.app.user_middleware) == 1
    assert client.get("/metrics").status_code == 404


def test_metrics_enabled_registers_middleware_after_timeout_middleware(
    prometheus_dir, sklearn_model_path
):
    client = _make_client(sklearn_model_path)

    # Starlette puts the most recently added middleware first (outermost)
    names = [m.kwargs["dispatch"].__name__ for m in client.app.user_middleware]
    assert names == ["prometheus_middleware", "timeout_middleware"]


def test_metrics_output_and_status_codes(prometheus_dir, sklearn_model_path):
    client = _make_client(sklearn_model_path)

    good = client.post("/invocations", content=GOOD_PAYLOAD, headers=JSON_HEADERS)
    assert good.status_code == 200
    bad = client.post("/invocations", content="{not json", headers=JSON_HEADERS)
    assert bad.status_code == 400

    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert REQUEST_COUNTER_NAME in response.text
    assert REQUEST_LATENCY_NAME in response.text

    assert _counter_values(client) == {
        ("/invocations", "200"): 1,
        ("/invocations", "400"): 1,
    }
    assert _histogram_counts(client) == {"/invocations": 2}


def test_metrics_scrape_is_not_counted(prometheus_dir, sklearn_model_path):
    client = _make_client(sklearn_model_path)

    for _ in range(3):
        assert client.get("/metrics").status_code == 200

    assert _counter_values(client) == {}
    assert _histogram_counts(client) == {}


def test_metrics_known_paths_are_labelled_by_path(prometheus_dir, sklearn_model_path):
    client = _make_client(sklearn_model_path)

    assert client.get("/ping").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/version").status_code == 200

    assert _counter_values(client) == {
        ("/ping", "200"): 1,
        ("/health", "200"): 1,
        ("/version", "200"): 1,
    }


def test_metrics_timeout_is_recorded_as_504(monkeypatch, prometheus_dir, sleep_model_path):
    monkeypatch.setenv("MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT", "1")
    client = _make_client(sleep_model_path)

    response = client.post(
        "/invocations",
        content=json.dumps({"inputs": [[1.0, 2.0]]}),
        headers=JSON_HEADERS,
    )
    assert response.status_code == 504

    assert _counter_values(client) == {("/invocations", "504"): 1}
    assert _histogram_counts(client) == {"/invocations": 1}


def test_metrics_unknown_paths_are_bounded(prometheus_dir, sklearn_model_path):
    client = _make_client(sklearn_model_path)

    assert client.get("/does-not-exist").status_code == 404
    assert client.get("/random/123").status_code == 404

    assert _counter_values(client) == {(OTHER_PATH_LABEL, "404"): 2}
    assert _histogram_counts(client) == {OTHER_PATH_LABEL: 2}

    counter_samples = _samples(client, REQUEST_COUNTER_NAME)
    assert {frozenset(s.labels) for s in counter_samples} == {frozenset({"path", "status"})}
    histogram_samples = [
        s
        for name in ("_bucket", "_count", "_sum")
        for s in _samples(client, REQUEST_LATENCY_NAME + name)
    ]
    assert histogram_samples
    assert {frozenset(s.labels) - {"le"} for s in histogram_samples} == {frozenset({"path"})}


_CHILD_SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    from fastapi.testclient import TestClient

    import mlflow.pyfunc
    from mlflow.pyfunc import scoring_server

    model_path, num_requests = sys.argv[1], int(sys.argv[2])
    client = TestClient(scoring_server.init(mlflow.pyfunc.load_model(model_path)))
    for _ in range(num_requests):
        response = client.post(
            "/invocations",
            content=json.dumps({"inputs": [[1.0, 2.0]]}),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200, response.text
    """
)


def test_metrics_aggregate_across_processes(prometheus_dir, sklearn_model_path):
    num_requests = 3
    env = {k: v for k, v in os.environ.items() if k != "PROMETHEUS_MULTIPROC_DIR"}
    children = [
        subprocess.Popen(
            [sys.executable, "-c", _CHILD_SCRIPT, sklearn_model_path, str(num_requests)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for _ in range(2)
    ]
    for child in children:
        output, _ = child.communicate(timeout=300)
        assert child.returncode == 0, output

    # The scrape is served by a third process (this one) that handled no requests itself
    client = _make_client(sklearn_model_path)
    assert _counter_values(client) == {("/invocations", "200"): 2 * num_requests}
    assert _histogram_counts(client) == {"/invocations": 2 * num_requests}


def test_metrics_disabled_with_warning_when_prometheus_client_is_missing(
    monkeypatch, caplog, tmp_path, sklearn_model_path
):
    monkeypatch.setenv("prometheus_multiproc_dir", str(tmp_path))
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    monkeypatch.setitem(sys.modules, "prometheus_client", None)

    with caplog.at_level(logging.WARNING):
        client = _make_client(sklearn_model_path)

    assert any(
        "prometheus_client" in r.getMessage() and r.levelno == logging.WARNING
        for r in caplog.records
    )
    assert len(client.app.user_middleware) == 1
    assert (
        client.post("/invocations", content=GOOD_PAYLOAD, headers=JSON_HEADERS).status_code == 200
    )
    assert client.get("/metrics").status_code == 404
