import json
import os
from unittest import mock

import mlflow
from mlflow.utils.doctor import _get_doctor_info


def test_doctor(capsys):
    mlflow.doctor()
    captured = capsys.readouterr()
    assert f"MLflow version: {mlflow.__version__}" in captured.out


def test_doctor_active_run(capsys):
    with mlflow.start_run() as run:
        mlflow.doctor()
        captured = capsys.readouterr()
        assert f"Active run ID: {run.info.run_id}" in captured.out


def test_doctor_databricks_runtime(capsys):
    mock_version = "12.0"
    with mock.patch(
        "mlflow.utils.doctor.get_databricks_runtime_version", return_value=mock_version
    ) as mock_runtime:
        mlflow.doctor()
        mock_runtime.assert_called_once()
        captured = capsys.readouterr()
        assert f"Databricks runtime version: {mock_version}" in captured.out


def test_get_doctor_info_keys_and_order(monkeypatch):
    for k in [k for k in os.environ if k.startswith("MLFLOW_")]:
        monkeypatch.delenv(k)
    with mock.patch("mlflow.utils.doctor.get_databricks_runtime_version", return_value=None):
        info = _get_doctor_info()
    assert list(info) == [
        "System information",
        "Python version",
        "MLflow version",
        "MLflow module location",
        "Tracking URI",
        "Registry URI",
        "MLflow dependencies",
    ]
    assert info["MLflow version"] == mlflow.__version__
    assert isinstance(info["MLflow dependencies"], dict)
    json.dumps(info)


def test_get_doctor_info_optional_fields(monkeypatch):
    monkeypatch.setenv("MLFLOW_DOCTOR_TEST_VAR", "secret")
    with (
        mock.patch("mlflow.utils.doctor.get_databricks_runtime_version", return_value="12.0"),
        mlflow.start_run() as run,
    ):
        info = _get_doctor_info()
    assert info["Databricks runtime version"] == "12.0"
    assert info["Active experiment ID"] == run.info.experiment_id
    assert info["Active run ID"] == run.info.run_id
    assert info["Active run artifact URI"] == run.info.artifact_uri
    assert info["MLflow environment variables"]["MLFLOW_DOCTOR_TEST_VAR"] == "secret"
    json.dumps(info)


def test_get_doctor_info_mask_envs(monkeypatch):
    monkeypatch.setenv("MLFLOW_DOCTOR_TEST_VAR", "secret")
    info = _get_doctor_info(mask_envs=True)
    envs = info["MLflow environment variables"]
    assert envs["MLFLOW_DOCTOR_TEST_VAR"] == "***"
    assert set(envs.values()) == {"***"}
