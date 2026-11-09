import json
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

import mlflow
from mlflow.pyfunc import PythonModel, scoring_server
from mlflow.pyfunc.scoring_server import prometheus_metrics

from tests.helper_functions import pyfunc_scoring_endpoint

COUNTER_FAMILY = "mlflow_scoring_server_http_request"
COUNTER_SAMPLE = "mlflow_scoring_server_http_request_total"
HISTOGRAM_FAMILY = "mlflow_scoring_server_http_request_duration_seconds"
HISTOGRAM_COUNT = "mlflow_scoring_server_http_request_duration_seconds_count"
HISTOGRAM_SUM = "mlflow_scoring_server_http_request_duration_seconds_sum"
HISTOGRAM_BUCKET = "mlflow_scoring_server_http_request_duration_seconds_bucket"

JSON_HEADERS = {"Content-Type": "application/json"}
GOOD_PAYLOAD = json.dumps({"inputs": [1, 2, 3]})
ENV_VARS = ("prometheus_multiproc_dir", "PROMETHEUS_MULTIPROC_DIR")


# Models are defined inside factories so cloudpickle serializes them by value, which keeps them
# loadable in spawned processes and in `mlflow models serve` subprocesses.
def _make_echo_model():
    class EchoModel(PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    return EchoModel()


def _make_sleeping_model(seconds):
    class SleepingModel(PythonModel):
        def predict(self, context, model_input, params=None):
            time.sleep(seconds)
            return model_input

    return SleepingModel()


@pytest.fixture(scope="module")
def echo_model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("echo_model") / "model"
    mlflow.pyfunc.save_model(path=str(path), python_model=_make_echo_model())
    return str(path)


@pytest.fixture(scope="module")
def sleeping_model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("sleeping_model") / "model"
    mlflow.pyfunc.save_model(path=str(path), python_model=_make_sleeping_model(2))
    return str(path)


@pytest.fixture
def prometheus_multiproc_dir(tmp_path, monkeypatch):
    prometheus_client = pytest.importorskip("prometheus_client")
    import prometheus_client.values

    multiproc_dir = tmp_path / "prometheus_multiproc"
    multiproc_dir.mkdir()
    # prometheus_client may already be imported in this process without multiprocess mode, so
    # force it and give every test fresh mmap files.
    for name in ENV_VARS:
        monkeypatch.setenv(name, str(multiproc_dir))
    monkeypatch.setattr(
        prometheus_client.values, "ValueClass", prometheus_client.values.MultiProcessValue()
    )
    monkeypatch.setattr(prometheus_metrics, "_metrics", None)
    yield multiproc_dir
    close_all = getattr(prometheus_client.values, "close_all_multiprocess_files", None)
    if close_all is not None:
        close_all()


def _make_client(model_path):
    return TestClient(scoring_server.init(mlflow.pyfunc.load_model(model_path)))


def _scrape(client):
    from prometheus_client.parser import text_string_to_metric_families

    response = client.get("/metrics")
    assert response.status_code == 200
    return list(text_string_to_metric_families(response.text))


def _value(families, name, **labels):
    for family in families:
        for sample in family.samples:
            if sample.name == name and sample.labels == labels:
                return sample.value
    return 0


def _all_samples(families):
    return [sample for family in families for sample in family.samples]


def _invoke(client, data=GOOD_PAYLOAD, headers=JSON_HEADERS):
    return client.post("/invocations", content=data, headers=headers)


def test_disabled_without_env_var(echo_model_path, monkeypatch, caplog):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # Proves that no prometheus_client import happens when metrics are disabled
    monkeypatch.setitem(sys.modules, "prometheus_client", None)

    with caplog.at_level(logging.WARNING):
        app = scoring_server.init(mlflow.pyfunc.load_model(echo_model_path))
        client = TestClient(app)
        assert client.get("/metrics").status_code == 404
        assert _invoke(client).status_code == 200

    assert len(app.user_middleware) == 1
    assert not [r for r in caplog.records if "prometheus" in r.getMessage().lower()]


def test_first_scrape_includes_both_metrics_before_any_request(
    echo_model_path, prometheus_multiproc_dir
):
    client = _make_client(echo_model_path)

    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert COUNTER_SAMPLE in response.text
    assert HISTOGRAM_BUCKET in response.text

    from prometheus_client.parser import text_string_to_metric_families

    families = {f.name: f for f in text_string_to_metric_families(response.text)}
    assert families[COUNTER_FAMILY].type == "counter"
    assert families[HISTOGRAM_FAMILY].type == "histogram"
    families = list(families.values())
    assert _value(families, COUNTER_SAMPLE, path="/invocations", status="200") == 0
    assert _value(families, HISTOGRAM_COUNT, path="/invocations") == 0

    assert _invoke(client).status_code == 200
    families = _scrape(client)
    assert _value(families, COUNTER_SAMPLE, path="/invocations", status="200") == 1


def test_status_code_labels(echo_model_path, prometheus_multiproc_dir):
    client = _make_client(echo_model_path)

    assert _invoke(client).status_code == 200
    assert _invoke(client).status_code == 200
    assert _invoke(client, data="{not valid json").status_code == 400
    assert _invoke(client, headers={"Content-Type": "text/plain"}).status_code == 415

    families = _scrape(client)
    assert _value(families, COUNTER_SAMPLE, path="/invocations", status="200") == 2
    assert _value(families, COUNTER_SAMPLE, path="/invocations", status="400") == 1
    assert _value(families, COUNTER_SAMPLE, path="/invocations", status="415") == 1
    assert _value(families, HISTOGRAM_COUNT, path="/invocations") == 4


def test_timeout_is_recorded_as_504(sleeping_model_path, prometheus_multiproc_dir, monkeypatch):
    monkeypatch.setenv("MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT", "1")
    client = _make_client(sleeping_model_path)

    assert _invoke(client).status_code == 504

    families = _scrape(client)
    assert _value(families, COUNTER_SAMPLE, path="/invocations", status="504") == 1
    assert _value(families, COUNTER_SAMPLE, path="/invocations", status="200") == 0
    assert _value(families, HISTOGRAM_COUNT, path="/invocations") == 1


def test_label_cardinality_is_bounded(echo_model_path, prometheus_multiproc_dir):
    client = _make_client(echo_model_path)

    assert client.get("/foo").status_code == 404
    assert client.get("/a/b/c").status_code == 404
    # Starlette redirects the trailing-slash path (307) rather than returning 404
    trailing_slash = client.post(
        "/invocations/", content=GOOD_PAYLOAD, headers=JSON_HEADERS, follow_redirects=False
    )
    assert trailing_slash.status_code in (307, 404)
    assert client.get("/invocations").status_code == 405

    families = _scrape(client)
    samples = [s for s in _all_samples(families) if s.name.startswith("mlflow_scoring_server_")]
    assert samples
    for sample in samples:
        assert sample.labels["path"] in {"/invocations", "other"}
        if sample.name == COUNTER_SAMPLE:
            assert set(sample.labels) == {"path", "status"}
        elif sample.name == HISTOGRAM_BUCKET:
            assert set(sample.labels) == {"path", "le"}
        else:
            assert set(sample.labels) == {"path"}

    assert _value(families, COUNTER_SAMPLE, path="/invocations", status="405") == 1
    is_404 = trailing_slash.status_code == 404
    assert _value(families, COUNTER_SAMPLE, path="other", status="404") == (3 if is_404 else 2)
    assert _value(
        families, COUNTER_SAMPLE, path="other", status=str(trailing_slash.status_code)
    ) == (3 if is_404 else 1)
    assert _value(families, HISTOGRAM_COUNT, path="other") == 3
    assert _value(families, HISTOGRAM_COUNT, path="/invocations") == 1


@pytest.mark.parametrize("path", ["/ping", "/health", "/version", "/metrics"])
def test_excluded_paths_are_not_tracked(echo_model_path, prometheus_multiproc_dir, path):
    client = _make_client(echo_model_path)

    for _ in range(3):
        client.get(path)

    families = _scrape(client)
    excluded = {"/ping", "/health", "/version", "/metrics"}
    for sample in _all_samples(families):
        if sample.name.startswith("mlflow_scoring_server_"):
            assert sample.labels["path"] not in excluded
    assert _value(families, COUNTER_SAMPLE, path="/invocations", status="200") == 0
    assert _value(families, COUNTER_SAMPLE, path="other", status="404") == 0
    assert _value(families, HISTOGRAM_COUNT, path="/invocations") == 0
    assert _value(families, HISTOGRAM_COUNT, path="other") == 0


def _worker_process(model_path, multiproc_dir, num_requests):
    for name in ENV_VARS:
        os.environ[name] = multiproc_dir
    model = mlflow.pyfunc.load_model(model_path)
    client = TestClient(scoring_server.init(model))
    for _ in range(num_requests):
        response = client.post("/invocations", content=GOOD_PAYLOAD, headers=JSON_HEADERS)
        assert response.status_code == 200
    sys.exit(0)


def test_multiple_processes_are_aggregated(echo_model_path, prometheus_multiproc_dir):
    num_processes = 2
    num_requests = 3
    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(
            target=_worker_process,
            args=(echo_model_path, str(prometheus_multiproc_dir), num_requests),
        )
        for _ in range(num_processes)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
        assert p.exitcode == 0

    client = _make_client(echo_model_path)
    families = _scrape(client)
    assert _value(families, COUNTER_SAMPLE, path="/invocations", status="200") == (
        num_processes * num_requests
    )
    assert _value(families, HISTOGRAM_COUNT, path="/invocations") == num_processes * num_requests

    own_file = f"counter_{os.getpid()}.db"
    other_files = {
        f.name for f in Path(prometheus_multiproc_dir).glob("counter_*.db") if f.name != own_file
    }
    assert len(other_files) >= num_processes


def test_multiple_workers_end_to_end(echo_model_path, tmp_path, monkeypatch):
    pytest.importorskip("prometheus_client")
    from prometheus_client.parser import text_string_to_metric_families

    multiproc_dir = tmp_path / "prometheus_multiproc"
    multiproc_dir.mkdir()
    monkeypatch.setenv("prometheus_multiproc_dir", str(multiproc_dir))

    num_requests = 10
    with pyfunc_scoring_endpoint(
        echo_model_path, extra_args=["--env-manager", "local", "--workers", "2"]
    ) as endpoint:
        for _ in range(num_requests):
            assert endpoint.invoke(GOOD_PAYLOAD, "application/json").status_code == 200
        response = requests.get(f"http://localhost:{endpoint._port}/metrics")

    assert response.status_code == 200
    families = list(text_string_to_metric_families(response.text))
    assert _value(families, COUNTER_SAMPLE, path="/invocations", status="200") == num_requests


def test_missing_prometheus_client_logs_warning(echo_model_path, tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("prometheus_multiproc_dir", str(tmp_path))
    monkeypatch.setitem(sys.modules, "prometheus_client", None)

    with caplog.at_level(logging.WARNING):
        app = scoring_server.init(mlflow.pyfunc.load_model(echo_model_path))

    assert any(
        r.levelno == logging.WARNING and "prometheus_client" in r.getMessage()
        for r in caplog.records
    )
    client = TestClient(app)
    assert _invoke(client).status_code == 200
    assert client.get("/metrics").status_code == 404
