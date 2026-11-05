# End-to-end tests for tag and alias conditions on registered-model role permissions.

import pytest
import requests

from mlflow.environment_variables import (
    MLFLOW_AUTH_CONFIG_PATH,
    MLFLOW_FLASK_SERVER_SECRET_KEY,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_USERNAME,
)
from mlflow.server.auth.permissions import EDIT, MANAGE, READ
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

from tests.helper_functions import random_str
from tests.server.auth.auth_test_utils import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    create_user,
    write_isolated_auth_config,
)
from tests.tracking.integration_test_utils import _init_server

ADMIN = (ADMIN_USERNAME, ADMIN_PASSWORD)
PROMOTED = {"type": "tag_equals", "key": "promoted", "value": "true"}
NOT_PROMOTED = {"type": "tag_not_equals", "key": "promoted", "value": "true"}


@pytest.fixture(autouse=True)
def clear_credentials(monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)


@pytest.fixture
def url(tmp_path, db_uri):
    auth_config_path = write_isolated_auth_config(tmp_path)
    with _init_server(
        backend_uri=db_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        app="mlflow.server.auth:create_app",
        extra_env={
            MLFLOW_FLASK_SERVER_SECRET_KEY.name: "my-secret-key",
            MLFLOW_AUTH_CONFIG_PATH.name: str(auth_config_path),
        },
        server_type="flask",
    ) as url:
        yield url


def _post(url, path, body, auth=ADMIN):
    return requests.post(f"{url}/api/2.0/mlflow/{path}", json=body, auth=auth)


def _grant(url, username, permission, condition=None, resource_type="registered_model"):
    role = requests.post(
        f"{url}/api/3.0/mlflow/roles/create",
        json={"name": f"r_{random_str()}", "workspace": DEFAULT_WORKSPACE_NAME},
        auth=ADMIN,
    )
    role.raise_for_status()
    role_id = role.json()["role"]["id"]
    body = {
        "role_id": role_id,
        "resource_type": resource_type,
        "resource_pattern": "*",
        "permission": permission,
    }
    if condition is not None:
        body["condition"] = condition
    requests.post(
        f"{url}/api/3.0/mlflow/roles/permissions/add", json=body, auth=ADMIN
    ).raise_for_status()
    requests.post(
        f"{url}/api/3.0/mlflow/roles/assign",
        json={"username": username, "role_id": role_id},
        auth=ADMIN,
    ).raise_for_status()


def _model(url):
    name = f"m_{random_str()}"
    _post(url, "registered-models/create", {"name": name}).raise_for_status()
    return name


def _get(url, name, auth):
    return requests.get(
        f"{url}/api/2.0/mlflow/registered-models/get", params={"name": name}, auth=auth
    ).status_code


def _update(url, name, auth):
    return requests.patch(
        f"{url}/api/2.0/mlflow/registered-models/update",
        json={"name": name, "description": "d"},
        auth=auth,
    ).status_code


def _delete(url, name, auth):
    return requests.delete(
        f"{url}/api/2.0/mlflow/registered-models/delete", json={"name": name}, auth=auth
    ).status_code


def _search_names(url, auth):
    resp = requests.get(
        f"{url}/api/2.0/mlflow/registered-models/search", params={"max_results": 1}, auth=auth
    )
    names = {m["name"] for m in resp.json().get("registered_models", [])}
    token = resp.json().get("next_page_token")
    while token:
        resp = requests.get(
            f"{url}/api/2.0/mlflow/registered-models/search",
            params={"max_results": 1, "page_token": token},
            auth=auth,
        )
        names |= {m["name"] for m in resp.json().get("registered_models", [])}
        token = resp.json().get("next_page_token")
    return names


def test_tag_conditions_follow_the_model_tags(url):
    user, password = create_user(url)
    auth = (user, password)
    _grant(url, user, EDIT.name, NOT_PROMOTED)
    _grant(url, user, READ.name, PROMOTED)
    name = _model(url)

    assert _update(url, name, auth) == 200
    assert _get(url, name, auth) == 200

    _post(url, "registered-models/set-tag", {"name": name, "key": "promoted", "value": "true"})
    assert _get(url, name, auth) == 200
    assert _update(url, name, auth) == 403
    assert _delete(url, name, auth) == 403
    assert name in _search_names(url, auth)

    requests.delete(
        f"{url}/api/2.0/mlflow/registered-models/delete-tag",
        json={"name": name, "key": "promoted"},
        auth=ADMIN,
    ).raise_for_status()
    assert _update(url, name, auth) == 200


def test_manage_with_tag_condition_gates_delete(url):
    user, password = create_user(url)
    _grant(url, user, MANAGE.name, NOT_PROMOTED)
    name = _model(url)
    tagged = _model(url)
    _post(url, "registered-models/set-tag", {"name": tagged, "key": "promoted", "value": "true"})

    assert _delete(url, tagged, (user, password)) == 403
    assert _delete(url, name, (user, password)) == 200


def test_conditional_deny_blocks_only_while_condition_holds(url):
    user, password = create_user(url)
    auth = (user, password)
    _grant(url, user, EDIT.name)
    _grant(url, user, "DENY", PROMOTED)
    plain = _model(url)
    tagged = _model(url)
    _post(url, "registered-models/set-tag", {"name": tagged, "key": "promoted", "value": "true"})

    assert _get(url, plain, auth) == 200
    assert _get(url, tagged, auth) == 403
    names = _search_names(url, auth)
    assert plain in names
    assert tagged not in names


def test_alias_conditions(url):
    user, password = create_user(url)
    auth = (user, password)
    _grant(url, user, EDIT.name, {"type": "alias_matches", "aliases": ["staging", "production-*"]})
    name = _model(url)
    exp_id = _post(url, "experiments/create", {"name": f"e_{random_str()}"}).json()["experiment_id"]
    run = _post(url, "runs/create", {"experiment_id": exp_id}).json()["run"]["info"]
    _post(
        url,
        "model-versions/create",
        {"name": name, "source": run["artifact_uri"], "run_id": run["run_id"]},
    ).raise_for_status()

    def set_alias(alias):
        return _post(
            url,
            "registered-models/alias",
            {"name": name, "alias": alias, "version": "1"},
            auth,
        ).status_code

    assert set_alias("production-eu") == 200
    assert set_alias("staging") == 200
    assert set_alias("champion") == 403
    delete = requests.delete(
        f"{url}/api/2.0/mlflow/registered-models/alias",
        json={"name": name, "alias": "staging"},
        auth=auth,
    )
    assert delete.status_code == 403
    # An alias condition governs nothing but set/delete alias.
    assert _update(url, name, auth) == 403


def test_invalid_conditions_are_rejected(url):
    role = requests.post(
        f"{url}/api/3.0/mlflow/roles/create",
        json={"name": f"r_{random_str()}", "workspace": DEFAULT_WORKSPACE_NAME},
        auth=ADMIN,
    ).json()["role"]
    for resource_type, condition in [
        ("registered_model", {"type": "bogus"}),
        ("registered_model", {"type": "alias_matches", "aliases": []}),
        ("experiment", PROMOTED),
    ]:
        resp = requests.post(
            f"{url}/api/3.0/mlflow/roles/permissions/add",
            json={
                "role_id": role["id"],
                "resource_type": resource_type,
                "resource_pattern": "*",
                "permission": "READ",
                "condition": condition,
            },
            auth=ADMIN,
        )
        assert resp.status_code == 400
