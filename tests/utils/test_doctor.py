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


def test_get_doctor_info_keys_and_order():
    info = _get_doctor_info()
    keys = list(info)
    assert keys[:6] == [
        "System information",
        "Python version",
        "MLflow version",
        "MLflow module location",
        "Tracking URI",
        "Registry URI",
    ]
    assert keys[-1] == "MLflow dependencies"
    assert isinstance(info["MLflow dependencies"], dict)
    assert info["MLflow version"] == mlflow.__version__
    json.dumps(info)


def test_get_doctor_info_optional_fields_omitted():
    with (
        mock.patch("mlflow.utils.doctor.get_databricks_runtime_version", return_value=None),
        mock.patch("mlflow.active_run", return_value=None),
        mock.patch.dict(os.environ, {}, clear=True),
    ):
        info = _get_doctor_info()
    assert "Databricks runtime version" not in info
    assert "Active run ID" not in info
    assert "MLflow environment variables" not in info


def test_get_doctor_info_active_run_and_runtime():
    with (
        mock.patch("mlflow.utils.doctor.get_databricks_runtime_version", return_value="12.0"),
        mlflow.start_run() as run,
    ):
        info = _get_doctor_info()
    assert info["Databricks runtime version"] == "12.0"
    assert info["Active experiment ID"] == run.info.experiment_id
    assert info["Active run ID"] == run.info.run_id
    assert info["Active run artifact URI"] == run.info.artifact_uri


def test_get_doctor_info_mask_envs():
    with mock.patch.dict(os.environ, {"MLFLOW_FOO": "secret"}):
        assert _get_doctor_info()["MLflow environment variables"]["MLFLOW_FOO"] == "secret"
        masked = _get_doctor_info(mask_envs=True)["MLflow environment variables"]
    assert masked["MLFLOW_FOO"] == "***"
    assert set(masked.values()) == {"***"}
