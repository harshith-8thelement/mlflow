from types import SimpleNamespace

import pytest
import requests
from flask import Flask

from mlflow import MlflowClient
from mlflow.environment_variables import MLFLOW_FLASK_SERVER_SECRET_KEY
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.protos.model_registry_pb2 import RegisteredModel as ProtoRegisteredModel
from mlflow.protos.model_registry_pb2 import RegisteredModelTag as ProtoRegisteredModelTag
from mlflow.server import auth as auth_module
from mlflow.server.auth import (
    _CONDITION_ALIAS,
    RetentionGate,
    _alias_condition_scope,
    _condition_context_for,
    _registered_model_tags_for_condition,
    _seed_registered_model_condition_tags,
)
from mlflow.server.auth.conditions import ConditionContext
from mlflow.server.auth.permissions import (
    DENY,
    EDIT,
    MANAGE,
    NO_PERMISSIONS,
    READ,
    RESOURCE_TYPE_EXPERIMENT,
    RESOURCE_TYPE_REGISTERED_MODEL,
    GrantLoadKey,
)
from mlflow.server.auth.requirements import Requirement
from mlflow.server.auth.sqlalchemy_store import RoleGrantRow, SqlAlchemyStore
from mlflow.utils.os import is_windows
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

from tests.helper_functions import random_str
from tests.server.auth.auth_test_utils import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    User,
    create_user,
    write_isolated_auth_config,
)
from tests.tracking.integration_test_utils import _init_server

NOT_PROMOTED = {"type": "tag_not_equals", "key": "promoted", "value": "true"}
PROMOTED = {"type": "tag_equals", "key": "promoted", "value": "true"}


def _alias_condition(*aliases):
    return {"type": "alias_matches", "aliases": list(aliases)}


@pytest.fixture
def server(tmp_path):
    config_path = write_isolated_auth_config(tmp_path)
    config_path.write_text(
        config_path.read_text().replace(
            "default_permission = READ", f"default_permission = {NO_PERMISSIONS.name}"
        )
    )
    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]
    with _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        extra_env={
            MLFLOW_FLASK_SERVER_SECRET_KEY.name: "my-secret-key",
            "MLFLOW_AUTH_CONFIG_PATH": str(config_path),
        },
        app="mlflow.server.auth:create_app",
        server_type="flask",
    ) as url:
        auth_store = SqlAlchemyStore()
        auth_store.init_db(f"sqlite:///{tmp_path / 'basic_auth.db'}")
        yield SimpleNamespace(client=MlflowClient(url), uri=url, store=auth_store)


def _new_user(server, *permissions):
    """A fresh user holding one role with ``(permission, condition)`` rows on all models."""
    username, password = create_user(server.uri)
    role = server.store.create_role(f"role-{random_str()}", DEFAULT_WORKSPACE_NAME)
    for permission, condition in permissions:
        server.store.add_role_permission(
            role.id,
            RESOURCE_TYPE_REGISTERED_MODEL,
            "*",
            permission,
            condition=condition,
        )
    server.store.assign_role_to_user(server.store.get_user(username).id, role.id)
    return username, password


@pytest.fixture
def admin(server, monkeypatch):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        yield server.client


def _post(server, path, body, user):
    return requests.post(f"{server.uri}/api/2.0/mlflow/{path}", json=body, auth=user, timeout=30)


def _get(server, path, params, user):
    return requests.get(f"{server.uri}/api/2.0/mlflow/{path}", params=params, auth=user, timeout=30)


def _tag(server, name, value, user):
    return _post(
        server, "registered-models/set-tag", {"name": name, "key": "promoted", "value": value}, user
    )


def _status_of_read(server, name, user):
    return _get(server, "registered-models/get", {"name": name}, user).status_code


def _status_of_update(server, name, user):
    return requests.patch(
        f"{server.uri}/api/2.0/mlflow/registered-models/update",
        json={"name": name, "description": "d"},
        auth=user,
        timeout=30,
    ).status_code


def _status_of_delete(server, name, user):
    return requests.delete(
        f"{server.uri}/api/2.0/mlflow/registered-models/delete",
        json={"name": name},
        auth=user,
        timeout=30,
    ).status_code


def test_tag_conditions_follow_the_models_current_tags(server, admin):
    user = _new_user(server, (EDIT.name, NOT_PROMOTED), (READ.name, PROMOTED))
    name = f"model-{random_str()}"
    admin.create_registered_model(name)

    assert _status_of_read(server, name, user) == 200
    assert _status_of_update(server, name, user) == 200

    admin.set_registered_model_tag(name, "promoted", "true")
    assert _status_of_read(server, name, user) == 200
    assert _status_of_update(server, name, user) == 403
    assert _status_of_delete(server, name, user) == 403
    assert _tag(server, name, "false", user).status_code == 403

    admin.delete_registered_model_tag(name, "promoted")
    assert _status_of_update(server, name, user) == 200


def test_tag_equals_edit_applies_only_while_tagged(server, admin):
    user = _new_user(server, (EDIT.name, PROMOTED))
    name = f"model-{random_str()}"
    admin.create_registered_model(name)
    assert _status_of_read(server, name, user) == 403

    admin.set_registered_model_tag(name, "promoted", "true")
    assert _status_of_update(server, name, user) == 200
    admin.set_registered_model_tag(name, "promoted", "false")
    assert _status_of_update(server, name, user) == 403


def test_conditional_deny_overrides_unconditional_grant(server, admin):
    user = _new_user(server, (MANAGE.name, None), (DENY.name, PROMOTED))
    name = f"model-{random_str()}"
    admin.create_registered_model(name)
    assert _status_of_update(server, name, user) == 200

    admin.set_registered_model_tag(name, "promoted", "true")
    assert _status_of_read(server, name, user) == 403
    assert _status_of_delete(server, name, user) == 403


def test_unconditional_permissions_are_unchanged(server, admin):
    user = _new_user(server, (EDIT.name, None))
    name = f"model-{random_str()}"
    admin.create_registered_model(name)
    admin.set_registered_model_tag(name, "promoted", "true")
    assert _status_of_update(server, name, user) == 200


def test_search_hides_models_a_tag_condition_makes_unreadable(server, admin):
    user = _new_user(server, (READ.name, PROMOTED))
    visible = f"visible-{random_str()}"
    hidden = f"hidden-{random_str()}"
    admin.create_registered_model(visible, tags={"promoted": "true"})
    admin.create_registered_model(hidden)

    resp = _get(server, "registered-models/search", {}, user)
    assert resp.status_code == 200
    assert [m["name"] for m in resp.json().get("registered_models", [])] == [visible]
    assert _status_of_read(server, visible, user) == 200
    assert _status_of_read(server, hidden, user) == 403

    admin.set_registered_model_tag(hidden, "promoted", "true")
    resp = _get(server, "registered-models/search", {}, user)
    assert {m["name"] for m in resp.json()["registered_models"]} == {visible, hidden}


def test_search_hides_models_when_not_equals_condition_stops_holding(server, admin):
    user = _new_user(server, (READ.name, NOT_PROMOTED))
    plain = f"plain-{random_str()}"
    promoted = f"promoted-{random_str()}"
    admin.create_registered_model(plain)
    admin.create_registered_model(promoted, tags={"promoted": "true"})

    resp = _get(server, "registered-models/search", {}, user)
    assert [m["name"] for m in resp.json().get("registered_models", [])] == [plain]
    assert _status_of_read(server, promoted, user) == 403


@pytest.fixture
def model_with_version(admin):
    name = f"model-{random_str()}"
    admin.create_registered_model(name)
    run = admin.create_run(admin.create_experiment(f"exp-{random_str()}"))
    admin.create_model_version(name, source=run.info.artifact_uri, run_id=run.info.run_id)
    return name


def _set_alias(server, name, alias, user):
    return _post(
        server,
        "registered-models/alias",
        {"name": name, "alias": alias, "version": "1"},
        user,
    ).status_code


def _delete_alias(server, name, alias, user):
    return requests.delete(
        f"{server.uri}/api/2.0/mlflow/registered-models/alias",
        json={"name": name, "alias": alias},
        auth=user,
        timeout=30,
    ).status_code


def test_set_alias_outside_the_list_is_denied(server, model_with_version):
    user = _new_user(server, (EDIT.name, _alias_condition("staging", "production-*")))
    name = model_with_version
    assert _set_alias(server, name, "champion", user) == 403
    assert _set_alias(server, name, "staging", user) == 200
    assert _set_alias(server, name, "production-eu", user) == 200
    assert _set_alias(server, name, "production", user) == 403
    assert _set_alias(server, name, "my-production-eu", user) == 403


def test_delete_alias_needs_manage_and_respects_the_list(server, admin, model_with_version):
    name = model_with_version
    for alias in ("champion", "production-eu"):
        admin.set_registered_model_alias(name, alias, "1")

    editor = _new_user(server, (EDIT.name, _alias_condition("production-*")))
    assert _delete_alias(server, name, "production-eu", editor) == 403

    manager = _new_user(server, (MANAGE.name, _alias_condition("production-*")))
    assert _delete_alias(server, name, "champion", manager) == 403
    assert _delete_alias(server, name, "production-eu", manager) == 200


def test_alias_condition_is_ignored_outside_alias_operations(server, admin):
    user = _new_user(server, (EDIT.name, _alias_condition("*")))
    name = f"model-{random_str()}"
    admin.create_registered_model(name)
    assert _status_of_read(server, name, user) == 403
    assert _status_of_update(server, name, user) == 403
    assert _tag(server, name, "true", user).status_code == 403


def test_alias_condition_combines_with_unconditional_grant(server, model_with_version):
    user = _new_user(server, (READ.name, None), (EDIT.name, _alias_condition("staging")))
    name = model_with_version
    assert _set_alias(server, name, "staging", user) == 200
    assert _set_alias(server, name, "other", user) == 403
    assert _status_of_update(server, name, user) == 403


def _grant(permission, pattern="*", condition=None):
    from mlflow.server.auth.conditions import serialize_condition

    return RoleGrantRow(
        RESOURCE_TYPE_REGISTERED_MODEL, pattern, permission, serialize_condition(condition)
    )


def _gate(grants, condition_context, default="NO_PERMISSIONS"):
    template = Requirement(RESOURCE_TYPE_REGISTERED_MODEL, "*", "read")
    return RetentionGate(
        {RESOURCE_TYPE_REGISTERED_MODEL: template},
        grants,
        default,
        NO_PERMISSIONS,
        condition_context=condition_context,
    )


def _tags_context(tags_by_name):
    return lambda key: ConditionContext(tags_by_name.get(key.resource_id), None)


def test_condition_context_is_empty_for_other_types_and_wildcards():
    empty = ConditionContext(None, None)
    assert _condition_context_for(GrantLoadKey(RESOURCE_TYPE_EXPERIMENT, "1")) == empty
    assert _condition_context_for(GrantLoadKey(RESOURCE_TYPE_REGISTERED_MODEL, "*")) == empty
    assert _condition_context_for(GrantLoadKey(RESOURCE_TYPE_REGISTERED_MODEL, None)) == empty


def test_condition_context_reads_tags_and_alias(monkeypatch):
    rm = SimpleNamespace(tags={"promoted": "true"})
    fetched = []
    registry = SimpleNamespace(get_registered_model=lambda name: fetched.append(name) or rm)
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: registry)

    key = GrantLoadKey(RESOURCE_TYPE_REGISTERED_MODEL, "m")
    assert _condition_context_for(key) == ConditionContext({"promoted": "true"}, None)
    with _alias_condition_scope("champion"):
        assert _condition_context_for(key) == ConditionContext({"promoted": "true"}, "champion")
    assert _CONDITION_ALIAS.get() is None


def test_missing_model_gives_no_tags(monkeypatch):
    def get_registered_model(name):
        raise MlflowException("nope", RESOURCE_DOES_NOT_EXIST)

    registry = SimpleNamespace(get_registered_model=get_registered_model)
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: registry)
    assert _registered_model_tags_for_condition("missing") is None


def test_tag_lookup_failure_propagates(monkeypatch):
    def get_registered_model(name):
        raise MlflowException("database is down")

    registry = SimpleNamespace(get_registered_model=get_registered_model)
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: registry)
    with pytest.raises(MlflowException, match="database is down"):
        _registered_model_tags_for_condition("m")


def test_tag_lookup_is_memoized_per_request(monkeypatch):
    calls = []

    def get_registered_model(name):
        calls.append(name)
        return SimpleNamespace(tags={"promoted": "true"})

    registry = SimpleNamespace(get_registered_model=get_registered_model)
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: registry)

    app = Flask(__name__)
    with app.test_request_context():
        assert _registered_model_tags_for_condition("m") == {"promoted": "true"}
        assert _registered_model_tags_for_condition("m") == {"promoted": "true"}
    assert calls == ["m"]

    _registered_model_tags_for_condition("m")
    _registered_model_tags_for_condition("m")
    assert calls == ["m", "m", "m"]


def test_seeding_prefills_the_cache_from_entities_and_protos(monkeypatch):
    registry = SimpleNamespace(
        get_registered_model=lambda name: pytest.fail("seeded tags should not be refetched")
    )
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: registry)

    proto = ProtoRegisteredModel(
        name="proto-model", tags=[ProtoRegisteredModelTag(key="promoted", value="true")]
    )
    entity = SimpleNamespace(name="entity-model", tags={"promoted": "false"})
    version_row = SimpleNamespace(name="version-model", version="1", tags={"promoted": "true"})

    app = Flask(__name__)
    with app.test_request_context():
        _seed_registered_model_condition_tags(proto)
        _seed_registered_model_condition_tags(entity)
        _seed_registered_model_condition_tags(version_row)
        assert _registered_model_tags_for_condition("proto-model") == {"promoted": "true"}
        assert _registered_model_tags_for_condition("entity-model") == {"promoted": "false"}
        with pytest.raises(pytest.fail.Exception, match="refetched"):
            _registered_model_tags_for_condition("version-model")


def test_seeding_outside_a_request_is_a_no_op():
    _seed_registered_model_condition_tags(SimpleNamespace(name="m", tags={"a": "b"}))


def test_retention_gate_applies_conditional_rows_per_resource():
    gate = _gate(
        [_grant("READ", condition=PROMOTED)],
        _tags_context({"a": {"promoted": "true"}, "b": {}}),
    )
    assert gate.retains(RESOURCE_TYPE_REGISTERED_MODEL, "a")
    assert not gate.retains(RESOURCE_TYPE_REGISTERED_MODEL, "b")
    assert not gate.retains(RESOURCE_TYPE_REGISTERED_MODEL, "unknown")


def test_retention_gate_skips_conditional_rows_without_a_provider():
    gate = _gate([_grant("READ", condition=PROMOTED)], None)
    assert not gate.retains(RESOURCE_TYPE_REGISTERED_MODEL, "a")


def test_retention_gate_conditional_deny_wins_over_unconditional_grant():
    gate = _gate(
        [_grant("MANAGE"), _grant("DENY", condition=PROMOTED)],
        _tags_context({"a": {"promoted": "true"}, "b": {}}),
    )
    assert not gate.retains(RESOURCE_TYPE_REGISTERED_MODEL, "a")
    assert gate.retains(RESOURCE_TYPE_REGISTERED_MODEL, "b")


def test_retention_gate_combines_conditional_and_named_with_max():
    gate = _gate(
        [_grant("NO_PERMISSIONS", pattern="a"), _grant("READ", condition=PROMOTED)],
        _tags_context({"a": {"promoted": "true"}}),
    )
    assert gate.retains(RESOURCE_TYPE_REGISTERED_MODEL, "a")


def test_retention_gate_denies_when_the_context_cannot_be_built():
    def broken(key):
        raise RuntimeError("store unavailable")

    gate = _gate([_grant("MANAGE"), _grant("READ", condition=PROMOTED)], broken)
    assert not gate.retains(RESOURCE_TYPE_REGISTERED_MODEL, "a")


def test_resolve_permissions_fails_closed_on_evaluation_error(monkeypatch):
    grants = [_grant("EDIT"), _grant("READ", condition=PROMOTED)]
    monkeypatch.setattr(
        auth_module,
        "store",
        SimpleNamespace(
            get_user=lambda username: SimpleNamespace(id=1),
            list_grants=lambda *args: grants,
        ),
    )

    def broken(name):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(auth_module, "_registered_model_tags_for_condition", broken)
    key = GrantLoadKey(RESOURCE_TYPE_REGISTERED_MODEL, "m")
    assert auth_module.resolve_permissions("u", DEFAULT_WORKSPACE_NAME, [key]) == [DENY]


def test_role_grant_for_resource_uses_conditions(monkeypatch):
    grants = [_grant("EDIT", condition=NOT_PROMOTED), _grant("READ", condition=PROMOTED)]
    monkeypatch.setattr(auth_module, "store", SimpleNamespace(list_grants=lambda *args: grants))
    tags = {"m": {"promoted": "true"}, "n": {}}
    monkeypatch.setattr(
        auth_module, "_registered_model_tags_for_condition", lambda name: tags.get(name)
    )
    role_grant = auth_module._role_grant_for_resource
    assert role_grant(1, RESOURCE_TYPE_REGISTERED_MODEL, "m", DEFAULT_WORKSPACE_NAME) == READ
    assert role_grant(1, RESOURCE_TYPE_REGISTERED_MODEL, "n", DEFAULT_WORKSPACE_NAME) == EDIT
    # An unknown model has no tags, so neither tag condition holds and the role stays silent.
    assert role_grant(1, RESOURCE_TYPE_REGISTERED_MODEL, "gone", DEFAULT_WORKSPACE_NAME) is None
