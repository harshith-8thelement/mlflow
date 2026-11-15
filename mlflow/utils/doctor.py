import os
import platform
from typing import Any

import click
import importlib_metadata
import yaml
from packaging.requirements import Requirement

import mlflow
from mlflow.utils.databricks_utils import get_databricks_runtime_version


def _get_doctor_info(mask_envs: bool = False) -> dict[str, Any]:
    """Collect the doctor debug info as an insertion-ordered dict."""
    info: dict[str, Any] = {
        "System information": " ".join((platform.system(), platform.version())),
        "Python version": platform.python_version(),
        "MLflow version": mlflow.__version__,
        "MLflow module location": mlflow.__file__,
        "Tracking URI": mlflow.get_tracking_uri(),
        "Registry URI": mlflow.get_registry_uri(),
    }

    if (runtime := get_databricks_runtime_version()) is not None:
        info["Databricks runtime version"] = runtime

    if active_run := mlflow.active_run():
        info["Active experiment ID"] = active_run.info.experiment_id
        info["Active run ID"] = active_run.info.run_id
        info["Active run artifact URI"] = active_run.info.artifact_uri

    mlflow_envs = {
        k: ("***" if mask_envs else v) for k, v in os.environ.items() if k.startswith("MLFLOW_")
    }
    if mlflow_envs:
        info["MLflow environment variables"] = mlflow_envs

    try:
        requires = importlib_metadata.requires("mlflow")
    except importlib_metadata.PackageNotFoundError:
        requires = importlib_metadata.requires("mlflow-skinny")

    mlflow_dependencies = {}
    for req in requires:
        req = Requirement(req)
        try:
            dist = importlib_metadata.distribution(req.name)
        except importlib_metadata.PackageNotFoundError:
            continue
        else:
            mlflow_dependencies[req.name] = dist.version

    info["MLflow dependencies"] = mlflow_dependencies
    return info


def doctor(mask_envs=False):
    """Prints out useful information for debugging issues with MLflow.

    Args:
        mask_envs: If True, mask the MLflow environment variable values
            (e.g. `"MLFLOW_ENV_VAR": "***"`) in the output to prevent leaking sensitive
            information.

    .. warning::

        - This API should only be used for debugging purposes.
        - The output may contain sensitive information such as a database URI containing a password.

    .. code-block:: python
        :caption: Example

        import mlflow

        with mlflow.start_run():
            mlflow.doctor()

    .. code-block:: text
        :caption: Output

        System information: Linux #58~20.04.1-Ubuntu SMP Thu Oct 13 13:09:46 UTC 2022
        Python version: 3.8.13
        MLflow version: 2.0.1
        MLflow module location: /usr/local/lib/python3.8/site-packages/mlflow/__init__.py
        Tracking URI: sqlite:///mlflow.db
        Registry URI: sqlite:///mlflow.db
        MLflow environment variables:
          MLFLOW_TRACKING_URI: sqlite:///mlflow.db
        MLflow dependencies:
          Flask: 2.2.2
          Jinja2: 3.0.3
          alembic: 1.8.1
          click: 8.1.3
          cloudpickle: 2.2.0
          databricks-cli: 0.17.4.dev0
          docker: 6.0.0
          entrypoints: 0.4
          gitpython: 3.1.29
          gunicorn: 20.1.0
          importlib-metadata: 5.0.0
          markdown: 3.4.1
          matplotlib: 3.6.1
          numpy: 1.23.4
          packaging: 21.3
          pandas: 1.5.1
          protobuf: 3.19.6
          pyarrow: 9.0.0
          pytz: 2022.6
          pyyaml: 6.0
          querystring-parser: 1.2.4
          requests: 2.28.1
          scikit-learn: 1.1.3
          scipy: 1.9.3
          shap: 0.41.0
          sqlalchemy: 1.4.42
          sqlparse: 0.4.3
    """
    for key, val in _get_doctor_info(mask_envs).items():
        if isinstance(val, dict):
            val = yaml.dump({"_": val}, indent=2).replace("'", "").lstrip("_:").rstrip("\n")
        click.secho(key, fg="blue", nl=False)
        click.echo(f": {val}")
