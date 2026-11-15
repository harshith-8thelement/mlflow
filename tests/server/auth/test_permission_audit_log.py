import pytest
import requests

from mlflow import MlflowClient, MlflowException
from mlflow.environment_variables import (
    MLFLOW_AUTH_CONFIG_PATH,
    MLFLOW_FLASK_SERVER_SECRET_KEY,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_USERNAME,
)
from mlflow.protos.databricks_pb2 import PERMISSION_DENIED, ErrorCode
from mlflow.server.auth.client import AuthServiceClient
from mlflow.server.auth.entities import PermissionAuditLogEntry
from mlflow.utils.os import is_windows

from tests.helper_functions import random_str
from tests.server.auth.auth_test_utils import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    User,
    write_isolated_auth_config,
)
from tests.tracking.integration_test_utils import _init_server


@pytest.fixture(autouse=True)
def clear_credentials(monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)


@pytest.fixture
def client(tmp_path):
    auth_config_path = write_isolated_auth_config(tmp_path)
    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]

    with _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        app="mlflow.server.auth:create_app",
        extra_env={
            MLFLOW_FLASK_SERVER_SECRET_KEY.name: "my-secret-key",
            MLFLOW_AUTH_CONFIG_PATH.name: str(auth_config_path),
        },
        server_type="flask",
    ) as url:
        yield AuthServiceClient(url)


def _new_user(client, monkeypatch):
    username = random_str()
    password = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_user(username, password)
    return username, password


def _create_experiment(tracking_uri, monkeypatch, name):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        return MlflowClient(tracking_uri).create_experiment(name)


def _audit_url(client, prefix="api"):
    return f"{client.tracking_uri}/{prefix}/2.0/mlflow/permissions/audit-log"


def test_grant_creates_one_create_entry(client, monkeypatch):
    username, _ = _new_user(client, monkeypatch)
    exp_id = _create_experiment(client.tracking_uri, monkeypatch, f"audit-{random_str()}")
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.grant_user_permission(username, "experiment", exp_id, "READ")
        entries = client.list_permission_audit_log(username=username)

    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, PermissionAuditLogEntry)
    assert entry.action == "create"
    assert entry.actor_username == ADMIN_USERNAME
    assert entry.target_username == username
    assert entry.resource_type == "experiment"
    assert entry.resource_id == exp_id
    assert entry.old_permission is None
    assert entry.new_permission == "READ"
    assert entry.timestamp > 0


def test_revoke_creates_one_delete_entry(client, monkeypatch):
    username, _ = _new_user(client, monkeypatch)
    exp_id = _create_experiment(client.tracking_uri, monkeypatch, f"audit-{random_str()}")
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.grant_user_permission(username, "experiment", exp_id, "EDIT")
        client.revoke_user_permission(username, "experiment", exp_id)
        entries = client.list_permission_audit_log(username=username)

    assert [e.action for e in entries] == ["delete", "create"]
    delete_entry = entries[0]
    assert delete_entry.actor_username == ADMIN_USERNAME
    assert delete_entry.target_username == username
    assert delete_entry.resource_type == "experiment"
    assert delete_entry.resource_id == exp_id
    assert delete_entry.old_permission == "EDIT"
    assert delete_entry.new_permission is None


def test_actor_is_the_acting_non_admin_manager(client, monkeypatch):
    target, _ = _new_user(client, monkeypatch)
    manager, manager_password = _new_user(client, monkeypatch)
    exp_id = _create_experiment(client.tracking_uri, monkeypatch, f"audit-{random_str()}")
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.grant_user_permission(manager, "experiment", exp_id, "MANAGE")
    with User(manager, manager_password, monkeypatch):
        client.grant_user_permission(target, "experiment", exp_id, "READ")
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        entries = client.list_permission_audit_log(username=target)

    assert len(entries) == 1
    assert entries[0].actor_username == manager


def test_failed_grants_and_revokes_add_no_entry(client, monkeypatch):
    username, _ = _new_user(client, monkeypatch)
    exp_id = _create_experiment(client.tracking_uri, monkeypatch, f"audit-{random_str()}")
    other_exp = _create_experiment(client.tracking_uri, monkeypatch, f"audit-{random_str()}")
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.grant_user_permission(username, "experiment", exp_id, "READ")
        before = client.list_permission_audit_log()

        with pytest.raises(MlflowException, match="already exists"):
            client.grant_user_permission(username, "experiment", exp_id, "EDIT")
        with pytest.raises(MlflowException, match="not found"):
            client.grant_user_permission("no-such-user-" + random_str(), "experiment", "1", "READ")
        with pytest.raises(MlflowException, match="Invalid resource type"):
            client.grant_user_permission(username, "bogus", "x", "READ")
        with pytest.raises(MlflowException, match="not found"):
            client.revoke_user_permission("no-such-user-" + random_str(), "experiment", "1")
        with pytest.raises(MlflowException, match="Invalid resource type"):
            client.revoke_user_permission(username, "bogus", "x")
        # The user has no grant on this experiment, so there is nothing to revoke.
        with pytest.raises(MlflowException, match=r".+"):
            client.revoke_user_permission(username, "experiment", other_exp)

        after = client.list_permission_audit_log()

    assert [e.id for e in after] == [e.id for e in before]


def test_entries_are_newest_first(client, monkeypatch):
    username, _ = _new_user(client, monkeypatch)
    exp_ids = [
        _create_experiment(client.tracking_uri, monkeypatch, f"audit-{random_str()}")
        for _ in range(3)
    ]
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        for exp_id in exp_ids:
            client.grant_user_permission(username, "experiment", exp_id, "READ")
        entries = client.list_permission_audit_log(username=username)

    assert [e.resource_id for e in entries] == exp_ids[::-1]


def test_username_and_resource_type_filters(client, monkeypatch):
    user_a, _ = _new_user(client, monkeypatch)
    user_b, _ = _new_user(client, monkeypatch)
    exp_id = _create_experiment(client.tracking_uri, monkeypatch, f"audit-{random_str()}")
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.grant_user_permission(user_a, "experiment", exp_id, "READ")
        client.grant_user_permission(user_b, "experiment", exp_id, "READ")
        client.grant_user_permission(user_a, "registered_model", "some-model", "READ")

        only_a = client.list_permission_audit_log(username=user_a)
        only_models = client.list_permission_audit_log(resource_type="registered_model")
        a_experiments = client.list_permission_audit_log(
            username=user_a, resource_type="experiment"
        )

    assert {e.target_username for e in only_a} == {user_a}
    assert len(only_a) == 2
    assert {e.resource_type for e in only_models} == {"registered_model"}
    assert [e.resource_id for e in only_models] == ["some-model"]
    assert [(e.target_username, e.resource_type) for e in a_experiments] == [(user_a, "experiment")]


def test_max_results_limits_entries(client, monkeypatch):
    username, _ = _new_user(client, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        for i in range(3):
            client.grant_user_permission(username, "registered_model", f"model-{i}", "READ")
        entries = client.list_permission_audit_log(username=username, max_results=2)

    assert [e.resource_id for e in entries] == ["model-2", "model-1"]


def test_default_limit_is_100(client, monkeypatch):
    username, _ = _new_user(client, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        for i in range(101):
            client.grant_user_permission(username, "registered_model", f"model-{i}", "READ")
        default = client.list_permission_audit_log(username=username)
        full = client.list_permission_audit_log(username=username, max_results=1000)

    assert len(default) == 100
    assert len(full) == 101


@pytest.mark.parametrize("max_results", ["1001", "0", "-1", "abc"])
def test_invalid_max_results_is_rejected(client, monkeypatch, max_results):
    resp = requests.get(
        _audit_url(client),
        params={"max_results": max_results},
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "max_results" in body["message"]


def test_non_admin_gets_403(client, monkeypatch):
    username, password = _new_user(client, monkeypatch)
    with User(username, password, monkeypatch):
        with pytest.raises(MlflowException, match="Permission denied") as exc:
            client.list_permission_audit_log()
    assert exc.value.error_code == ErrorCode.Name(PERMISSION_DENIED)

    resp = requests.get(_audit_url(client), auth=(username, password))
    assert resp.status_code == 403


def test_unauthenticated_request_is_rejected(client):
    resp = requests.get(_audit_url(client))
    assert resp.status_code == 401


@pytest.mark.parametrize("prefix", ["api", "ajax-api"])
def test_endpoint_reachable_at_both_path_prefixes(client, monkeypatch, prefix):
    username, password = _new_user(client, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.grant_user_permission(username, "registered_model", "m", "READ")

    resp = requests.get(
        _audit_url(client, prefix),
        params={"username": username},
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
    )
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    assert len(entries) == 1
    assert entries[0]["action"] == "create"

    denied = requests.get(_audit_url(client, prefix), auth=(username, password))
    assert denied.status_code == 403
