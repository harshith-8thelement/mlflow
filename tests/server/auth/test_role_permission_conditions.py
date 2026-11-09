"""Enforcement of role-permission conditions on registered models.

Runs the real validators and response filters against a real auth store and a real model
registry store, so the persisted tags, the requested alias and the stored conditions are what
the decisions are made from.
"""

import json
import uuid
from types import SimpleNamespace

import pytest
from flask import Response

from mlflow.entities.model_registry import RegisteredModelTag
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.protos.model_registry_pb2 import (
    SearchModelVersions,
    SearchRegisteredModels,
)
from mlflow.server import auth as auth_module
from mlflow.server.auth.conditions import (
    CONDITION_ALIAS_MATCHES,
    CONDITION_TAG_EQUALS,
    CONDITION_TAG_NOT_EQUALS,
    ConditionContext,
)
from mlflow.server.auth.db.models import SqlRolePermission
from mlflow.server.auth.permissions import (
    DENY,
    EDIT,
    MANAGE,
    NO_PERMISSIONS,
    READ,
    RESOURCE_TYPE_REGISTERED_MODEL,
    RESOURCE_TYPE_REGISTERED_MODEL_VERSION,
    RESOURCE_TYPE_WORKSPACE,
)
from mlflow.server.auth.requirements import Requirement
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore as RegistryStore
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

USERNAME = "alice"
MODEL = RESOURCE_TYPE_REGISTERED_MODEL

PROMOTED = {"type": CONDITION_TAG_EQUALS, "key": "promoted", "value": "true"}
NOT_PROMOTED = {"type": CONDITION_TAG_NOT_EQUALS, "key": "promoted", "value": "true"}
ALIASES = {"type": CONDITION_ALIAS_MATCHES, "aliases": ["staging", "production-*"]}
CHAMPION = {"type": CONDITION_ALIAS_MATCHES, "aliases": ["champion"]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission=NO_PERMISSIONS.name),
    )
    auth_store = SqlAlchemyStore()
    auth_store.init_db(f"sqlite:///{tmp_path / 'auth.db'}")
    monkeypatch.setattr(auth_module, "store", auth_store, raising=False)
    auth_store.create_user(USERNAME, "supersecurepassword", is_admin=False)

    registry = RegistryStore(f"sqlite:///{tmp_path / 'registry.db'}")
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: registry)
    monkeypatch.setattr(
        auth_module, "authenticate_request", lambda: SimpleNamespace(username=USERNAME)
    )
    yield SimpleNamespace(auth=auth_store, registry=registry)
    auth_store.engine.dispose()


def grant(env, rows, username=USERNAME):
    """Give ``username`` a fresh role holding ``(type, pattern, permission[, condition])`` rows."""
    role = env.auth.create_role(f"role-{uuid.uuid4().hex[:8]}", DEFAULT_WORKSPACE_NAME)
    for resource_type, pattern, permission, *condition in rows:
        env.auth.add_role_permission(
            role.id,
            resource_type,
            pattern,
            permission,
            condition=condition[0] if condition else None,
        )
    env.auth.assign_role_to_user(env.auth.get_user(username).id, role.id)
    return role


def make_model(env, name="m", tags=None):
    env.registry.create_registered_model(name)
    for key, value in (tags or {}).items():
        env.registry.set_registered_model_tag(name, _tag(key, value))
    return name


def _tag(key, value):
    return RegisteredModelTag(key, value)


def tag(env, name, key, value):
    env.registry.set_registered_model_tag(name, _tag(key, value))


def untag(env, name, key):
    env.registry.delete_registered_model_tag(name, key)


def check(validator, path, body, method="POST"):
    with auth_module.app.test_request_context(path, method=method, json=body):
        return bool(validator())


def can_read(name):
    return check(
        auth_module._validate_can_read_registered_model_or_prompt,
        "/api/2.0/mlflow/registered-models/get",
        {"name": name},
    )


def can_update(name):
    return check(
        auth_module._validate_can_update_registered_model_or_prompt,
        "/api/2.0/mlflow/registered-models/update",
        {"name": name},
    )


def can_set_tag(name):
    return check(
        auth_module._validate_can_update_registered_model_or_prompt,
        "/api/2.0/mlflow/registered-models/set-tag",
        {"name": name, "key": "k", "value": "v"},
    )


def can_delete_tag(name):
    return check(
        auth_module._validate_can_update_registered_model_or_prompt,
        "/api/2.0/mlflow/registered-models/delete-tag",
        {"name": name, "key": "k"},
    )


def can_delete(name):
    return check(
        auth_module.validate_can_delete_registered_model_or_prompt_cascade,
        "/api/2.0/mlflow/registered-models/delete",
        {"name": name},
        method="DELETE",
    )


def can_rename(name):
    return check(
        auth_module.validate_can_rename_registered_model,
        "/api/2.0/mlflow/registered-models/rename",
        {"name": name, "new_name": f"{name}-renamed"},
    )


def can_create_version(name):
    return check(
        auth_module.validate_can_create_model_version,
        "/api/2.0/mlflow/model-versions/create",
        {"name": name, "source": "s3://bucket/path"},
    )


def can_update_version(name):
    return check(
        auth_module.validate_can_update_model_or_prompt_version,
        "/api/2.0/mlflow/model-versions/update",
        {"name": name, "version": "1"},
    )


def can_delete_version(name):
    return check(
        auth_module.validate_can_delete_model_or_prompt_version,
        "/api/2.0/mlflow/model-versions/delete",
        {"name": name, "version": "1"},
        method="DELETE",
    )


def can_read_version(name):
    return check(
        auth_module.validate_can_read_model_or_prompt_version,
        "/api/2.0/mlflow/model-versions/get",
        {"name": name, "version": "1"},
    )


def can_set_alias(name, alias):
    return check(
        auth_module.validate_can_set_model_or_prompt_version_alias,
        "/api/2.0/mlflow/registered-models/alias",
        {"name": name, "alias": alias, "version": "1"},
    )


def can_delete_alias(name, alias):
    return check(
        auth_module.validate_can_delete_model_or_prompt_version_alias,
        "/api/2.0/mlflow/registered-models/alias",
        {"name": name, "alias": alias},
        method="DELETE",
    )


def can_create_model(name, tags=()):
    return check(
        auth_module.validate_can_create_registered_model,
        "/api/2.0/mlflow/registered-models/create",
        {"name": name, "tags": [{"key": k, "value": v} for k, v in tags]},
    )


MODIFYING = [
    can_update,
    can_set_tag,
    can_delete_tag,
    can_delete,
    can_rename,
    can_create_version,
    can_update_version,
    can_delete_version,
]


# ------------------------------------------------------------------ tag conditions


def test_tag_conditions_follow_the_models_current_tags(env):
    make_model(env)
    grant(env, [(MODEL, "*", MANAGE.name, NOT_PROMOTED), (MODEL, "*", READ.name, PROMOTED)])

    assert can_read("m")
    assert all(check_modifies("m") for check_modifies in MODIFYING)

    tag(env, "m", "promoted", "true")
    assert can_read("m")
    assert can_read_version("m")
    assert not any(check_modifies("m") for check_modifies in MODIFYING)

    untag(env, "m", "promoted")
    assert all(check_modifies("m") for check_modifies in MODIFYING)

    tag(env, "m", "promoted", "false")
    assert all(check_modifies("m") for check_modifies in MODIFYING)


def test_both_entries_on_the_same_pattern_apply_together(env):
    make_model(env, "a")
    make_model(env, "b", {"promoted": "true"})
    grant(env, [(MODEL, "*", EDIT.name, NOT_PROMOTED), (MODEL, "*", READ.name, PROMOTED)])

    assert can_update("a") and can_read("a")
    assert can_read("b") and not can_update("b")


def test_conditions_apply_to_named_grants_too(env):
    make_model(env, "a")
    make_model(env, "b")
    grant(env, [(MODEL, "a", EDIT.name, NOT_PROMOTED)])
    assert can_update("a")
    assert not can_update("b")
    tag(env, "a", "promoted", "true")
    assert not can_update("a")


def test_conditional_deny_vetoes_only_while_it_holds(env):
    make_model(env)
    grant(env, [(MODEL, "*", MANAGE.name), (MODEL, "*", DENY.name, PROMOTED)])
    assert can_update("m") and can_delete("m")
    tag(env, "m", "promoted", "true")
    assert not can_read("m")
    assert not can_update("m")
    assert not can_delete("m")


def test_a_second_role_widens_what_a_condition_narrows(env):
    make_model(env, tags={"promoted": "true"})
    grant(env, [(MODEL, "*", EDIT.name, NOT_PROMOTED)])
    assert not can_read("m")
    grant(env, [(MODEL, "*", READ.name)])
    assert can_read("m") and not can_update("m")


def test_version_routes_follow_parent_tags(env):
    make_model(env)
    grant(env, [(MODEL, "*", EDIT.name, NOT_PROMOTED)])
    assert can_read_version("m") and can_update_version("m") and can_create_version("m")
    tag(env, "m", "promoted", "true")
    assert not can_read_version("m")
    assert not can_update_version("m")
    assert not can_create_version("m")


def test_version_tier_veto_still_applies_under_a_condition(env):
    make_model(env)
    grant(
        env,
        [
            (MODEL, "*", EDIT.name, NOT_PROMOTED),
            (RESOURCE_TYPE_REGISTERED_MODEL_VERSION, "*", DENY.name),
        ],
    )
    assert can_read("m")
    assert not can_read_version("m")
    assert not can_create_version("m")


def test_unconditional_permissions_behave_as_before(env):
    make_model(env, tags={"promoted": "true"})
    grant(env, [(MODEL, "*", EDIT.name)])
    assert can_read("m") and can_update("m") and can_create_version("m")
    assert can_update_version("m")
    assert not can_delete("m")
    assert can_set_alias("m", "anything")
    assert not can_delete_alias("m", "anything")


def test_unconditional_manage_can_delete_any_alias(env):
    make_model(env)
    grant(env, [(MODEL, "*", MANAGE.name)])
    assert can_set_alias("m", "champion") and can_delete_alias("m", "champion")


def test_missing_model_never_satisfies_a_positive_tag_condition(env):
    grant(env, [(MODEL, "*", EDIT.name, NOT_PROMOTED)])
    assert not can_read("ghost")
    assert not can_update("ghost")
    assert not can_create_version("ghost")


def test_missing_model_is_conservative_for_a_conditional_deny(env):
    grant(env, [(MODEL, "*", MANAGE.name), (MODEL, "*", DENY.name, PROMOTED)])
    assert not can_read("ghost")


# ------------------------------------------------------------------ alias conditions


@pytest.mark.parametrize(
    ("alias", "allowed"),
    [
        ("staging", True),
        ("production-eu", True),
        ("production-", True),
        ("production", False),
        ("my-production-1", False),
        ("champion", False),
        ("Staging", False),
    ],
)
def test_alias_matches_gates_set_and_delete_by_alias(env, alias, allowed):
    make_model(env)
    grant(env, [(MODEL, "*", EDIT.name, ALIASES)])
    assert can_set_alias("m", alias) is allowed
    assert can_delete_alias("m", alias) is allowed


def test_alias_matches_grants_nothing_for_other_actions(env):
    make_model(env)
    grant(env, [(MODEL, "*", EDIT.name, ALIASES)])
    assert not can_read("m")
    assert not can_update("m")
    assert not can_set_tag("m")
    assert not can_create_version("m")
    assert not can_delete("m")
    assert not can_read_version("m")


def test_alias_matches_does_not_widen_a_lower_baseline(env):
    make_model(env)
    grant(env, [(MODEL, "*", READ.name), (MODEL, "*", EDIT.name, ALIASES)])
    assert can_read("m")
    assert not can_update("m")
    assert can_set_alias("m", "staging")
    assert not can_set_alias("m", "champion")
    assert can_delete_alias("m", "staging")
    assert not can_delete_alias("m", "champion")


def test_alias_matches_below_edit_cannot_set_or_delete(env):
    make_model(env)
    grant(env, [(MODEL, "*", READ.name, ALIASES)])
    assert not can_set_alias("m", "staging")
    assert not can_delete_alias("m", "staging")


def test_plain_edit_cannot_delete_an_alias_without_a_matching_condition(env):
    make_model(env)
    grant(env, [(MODEL, "*", EDIT.name), (MODEL, "*", READ.name, ALIASES)])
    assert can_set_alias("m", "champion")
    assert not can_delete_alias("m", "champion")
    assert not can_delete_alias("m", "staging")


def test_alias_deny_vetoes_set_and_delete_for_that_alias_only(env):
    make_model(env)
    grant(env, [(MODEL, "*", MANAGE.name), (MODEL, "*", DENY.name, CHAMPION)])
    assert not can_set_alias("m", "champion")
    assert not can_delete_alias("m", "champion")
    assert can_set_alias("m", "staging")
    assert can_delete_alias("m", "staging")
    assert can_update("m")


def test_alias_deny_beats_an_alias_scoped_grant(env):
    make_model(env)
    grant(
        env,
        [(MODEL, "*", EDIT.name, ALIASES), (MODEL, "*", DENY.name, {**CHAMPION, "aliases": ["staging"]})],
    )
    assert not can_delete_alias("m", "staging")
    assert not can_set_alias("m", "staging")
    assert can_set_alias("m", "production-1")


def test_alias_conditions_combine_with_tag_conditions(env):
    make_model(env)
    grant(
        env,
        [(MODEL, "*", EDIT.name, ALIASES), (MODEL, "*", DENY.name, PROMOTED)],
    )
    assert can_set_alias("m", "staging")
    tag(env, "m", "promoted", "true")
    assert not can_set_alias("m", "staging")
    assert not can_delete_alias("m", "staging")


def test_alias_operations_need_the_version_tier_read(env):
    make_model(env)
    grant(
        env,
        [
            (MODEL, "*", EDIT.name, ALIASES),
            (RESOURCE_TYPE_REGISTERED_MODEL_VERSION, "*", DENY.name),
        ],
    )
    assert not can_set_alias("m", "staging")
    assert not can_delete_alias("m", "staging")


def test_alias_on_a_missing_model_is_denied(env):
    grant(env, [(MODEL, "*", MANAGE.name), (MODEL, "*", EDIT.name, ALIASES)])
    assert not can_set_alias("ghost", "staging")
    assert not can_delete_alias("ghost", "staging")


def test_alias_param_is_required(env):
    make_model(env)
    grant(env, [(MODEL, "*", MANAGE.name)])
    with pytest.raises(Exception, match="alias"):
        check(
            auth_module.validate_can_set_model_or_prompt_version_alias,
            "/api/2.0/mlflow/registered-models/alias",
            {"name": "m", "version": "1"},
        )


# ------------------------------------------------------------------ create


def test_create_not_denied_uses_the_request_tags(env):
    deny_prod = {"type": CONDITION_TAG_EQUALS, "key": "env", "value": "prod"}
    grant(env, [(MODEL, "*", DENY.name, deny_prod)])
    assert can_create_model("new", [("env", "dev")])
    assert can_create_model("new")
    assert not can_create_model("new", [("env", "prod")])


def test_create_not_denied_with_tag_not_equals_deny(env):
    require_owner = {"type": CONDITION_TAG_NOT_EQUALS, "key": "owner", "value": "ml"}
    grant(env, [(MODEL, "*", DENY.name, require_owner)])
    assert can_create_model("new", [("owner", "ml")])
    assert not can_create_model("new", [("owner", "other")])
    assert not can_create_model("new")


# ------------------------------------------------------------------ search / listing


def search_registered_models_response(env):
    rms = env.registry.search_registered_models(max_results=100)
    message = SearchRegisteredModels.Response()
    message.registered_models.extend(rm.to_proto() for rm in rms)
    return Response(message_to_json(message), mimetype="application/json")


def visible_models(env):
    resp = search_registered_models_response(env)
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/search", method="POST", json={"max_results": 100}
    ):
        auth_module.filter_search_registered_models(resp)
    return sorted(m["name"] for m in json.loads(resp.data).get("registered_models", []))


def test_search_hides_models_whose_tags_make_them_unreadable(env):
    make_model(env, "plain")
    make_model(env, "promoted", {"promoted": "true"})
    grant(env, [(MODEL, "*", READ.name, NOT_PROMOTED)])

    assert visible_models(env) == ["plain"]
    assert can_read("plain") and not can_read("promoted")

    tag(env, "plain", "promoted", "true")
    untag(env, "promoted", "promoted")
    assert visible_models(env) == ["promoted"]
    assert can_read("promoted") and not can_read("plain")


def test_search_matches_the_point_route_for_every_tag_state(env):
    states = {"none": {}, "yes": {"promoted": "true"}, "no": {"promoted": "false"}}
    for name, tags in states.items():
        make_model(env, name, tags)
    grant(
        env,
        [(MODEL, "*", EDIT.name, NOT_PROMOTED), (MODEL, "none", DENY.name, PROMOTED)],
    )
    grant(env, [(MODEL, "yes", READ.name, PROMOTED)])

    assert visible_models(env) == sorted(name for name in states if can_read(name))


def test_search_hides_everything_for_an_alias_only_grant(env):
    make_model(env, "a")
    grant(env, [(MODEL, "*", EDIT.name, ALIASES)])
    assert visible_models(env) == []


def test_unconditional_grants_list_every_model(env):
    make_model(env, "a")
    make_model(env, "b", {"promoted": "true"})
    grant(env, [(MODEL, "*", READ.name)])
    assert visible_models(env) == ["a", "b"]


def test_latest_versions_are_withheld_when_the_model_is_unreadable_by_tag(env):
    make_model(env, "plain")
    make_model(env, "promoted", {"promoted": "true"})
    for name in ("plain", "promoted"):
        env.registry.create_model_version(name, "s3://bucket/path")
    grant(env, [(MODEL, "*", READ.name, NOT_PROMOTED)])

    rms = env.registry.search_registered_models(max_results=100)
    message = SearchRegisteredModels.Response()
    message.registered_models.extend(rm.to_proto() for rm in rms)
    assert all(m.latest_versions for m in message.registered_models)

    withheld = auth_module._withhold_denied_latest_versions(message.registered_models, USERNAME)

    assert withheld
    remaining = {m.name: len(m.latest_versions) for m in message.registered_models}
    assert remaining == {"plain": 1, "promoted": 0}


def test_model_version_search_follows_the_parents_tags(env):
    make_model(env, "plain")
    make_model(env, "promoted", {"promoted": "true"})
    for name in ("plain", "promoted"):
        env.registry.create_model_version(name, "s3://bucket/path")
    grant(env, [(MODEL, "*", READ.name, NOT_PROMOTED)])

    versions = env.registry.search_model_versions(max_results=100)
    message = SearchModelVersions.Response()
    message.model_versions.extend(v.to_proto() for v in versions)
    resp = Response(message_to_json(message), mimetype="application/json")
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/model-versions/search", method="POST", json={"max_results": 100}
    ):
        auth_module.filter_search_model_versions(resp)

    assert [v["name"] for v in json.loads(resp.data)["model_versions"]] == ["plain"]


def test_version_predicate_reads_each_parent_once(env, monkeypatch):
    make_model(env, "m")
    for _ in range(3):
        env.registry.create_model_version("m", "s3://bucket/path")
    grant(env, [(MODEL, "*", READ.name, NOT_PROMOTED)])
    calls = []
    real_get = env.registry.get_registered_model
    monkeypatch.setattr(
        env.registry, "get_registered_model", lambda name: calls.append(name) or real_get(name)
    )

    can_read_version = auth_module._rm_or_prompt_version_read_predicate(USERNAME)
    versions = env.registry.search_model_versions(max_results=100)
    calls.clear()
    assert all(can_read_version(v) for v in versions)
    assert calls == ["m"]


def test_version_predicate_does_not_read_parents_without_conditional_grants(env, monkeypatch):
    make_model(env, "m")
    env.registry.create_model_version("m", "s3://bucket/path")
    grant(env, [(MODEL, "*", READ.name)])
    versions = env.registry.search_model_versions(max_results=100)
    monkeypatch.setattr(
        env.registry,
        "get_registered_model",
        lambda name: pytest.fail("parent fetched without conditional grants"),
    )
    can_read_version = auth_module._rm_or_prompt_version_read_predicate(USERNAME)
    assert all(can_read_version(v) for v in versions)


def test_version_predicate_treats_a_vanished_parent_as_unknown_tags(env, monkeypatch):
    make_model(env, "m")
    env.registry.create_model_version("m", "s3://bucket/path")
    grant(env, [(MODEL, "*", READ.name, NOT_PROMOTED)])
    version = env.registry.search_model_versions(max_results=100)[0]
    monkeypatch.setattr(auth_module, "_fetch_or_none", lambda fetch, resource_id: None)
    assert not auth_module._rm_or_prompt_version_read_predicate(USERNAME)(version)


def test_graphql_model_read_follows_tags(env):
    make_model(env)
    grant(env, [(MODEL, "*", READ.name, NOT_PROMOTED)])
    assert auth_module._graphql_can_read_model("m", USERNAME)
    tag(env, "m", "promoted", "true")
    assert not auth_module._graphql_can_read_model("m", USERNAME)


# ------------------------------------------------------------------ retention gate parity


TEMPLATE = Requirement(
    MODEL, "*", "read", fallback_if_no_grant=((RESOURCE_TYPE_WORKSPACE, "*"),)
)
CONTEXTS = [
    ConditionContext(tags={}),
    ConditionContext(tags={"promoted": "true"}),
    ConditionContext(tags={"promoted": "false"}),
    ConditionContext(tags=None),
    None,
]


@pytest.mark.parametrize(
    "rows",
    [
        [(MODEL, "*", READ.name, NOT_PROMOTED)],
        [(MODEL, "*", READ.name, PROMOTED)],
        [(MODEL, "*", READ.name), (MODEL, "m", DENY.name, PROMOTED)],
        [(MODEL, "*", DENY.name, PROMOTED), (MODEL, "m", MANAGE.name)],
        [(MODEL, "m", READ.name, NOT_PROMOTED), (MODEL, "*", READ.name, PROMOTED)],
        [(MODEL, "*", READ.name, ALIASES)],
        [(MODEL, "*", DENY.name, ALIASES), (MODEL, "*", READ.name)],
        [(MODEL, "*", MANAGE.name, NOT_PROMOTED), (MODEL, "*", DENY.name)],
        [(MODEL, "other", READ.name)],
    ],
)
def test_retention_gate_matches_the_point_route_fold(env, rows):
    grant(env, rows)
    gate = auth_module.retention_gate(USERNAME, (RESOURCE_TYPE_WORKSPACE, "*"), [TEMPLATE])
    for _ in range(2):
        for context in CONTEXTS:
            expected = auth_module.authorize(
                USERNAME,
                (RESOURCE_TYPE_WORKSPACE, "*"),
                [TEMPLATE._replace(resource_id="m")],
                context,
            )
            assert gate.retains(MODEL, "m", context) is expected, context


def test_retention_gate_conditional_answers_are_not_memoised_across_contexts(env):
    grant(env, [(MODEL, "*", READ.name, NOT_PROMOTED)])
    gate = auth_module.retention_gate(USERNAME, (RESOURCE_TYPE_WORKSPACE, "*"), [TEMPLATE])
    assert gate.is_conditional(MODEL)
    assert gate.retains(MODEL, "m", ConditionContext(tags={}))
    assert not gate.retains(MODEL, "m", ConditionContext(tags={"promoted": "true"}))
    assert gate.retains(MODEL, "m", ConditionContext(tags={}))
    assert not gate._decided


def test_retention_gate_still_memoises_unconditional_types(env):
    grant(env, [(MODEL, "*", READ.name)])
    gate = auth_module.retention_gate(USERNAME, (RESOURCE_TYPE_WORKSPACE, "*"), [TEMPLATE])
    assert not gate.is_conditional(MODEL)
    assert gate.retains(MODEL, "m")
    assert (MODEL, "m") in gate._decided


def test_retention_gate_for_a_workspace_admin_ignores_conditions(env):
    grant(env, [(RESOURCE_TYPE_WORKSPACE, "*", MANAGE.name), (MODEL, "*", DENY.name, PROMOTED)])
    gate = auth_module.retention_gate(USERNAME, (RESOURCE_TYPE_WORKSPACE, "*"), [TEMPLATE])
    assert gate.retains(MODEL, "m", ConditionContext(tags={"promoted": "true"}))
    assert not gate.is_conditional(MODEL)


# ------------------------------------------------------------------ fail closed


def corrupt_conditions(env, value):
    with env.auth.ManagedSessionMaker(read_only=False) as session:
        for rp in session.query(SqlRolePermission).filter(SqlRolePermission.condition.isnot(None)):
            rp.condition = value


@pytest.mark.parametrize(
    "corrupted",
    [
        "not json",
        "[1, 2]",
        '{"type": "who_knows"}',
        '{"type": "tag_equals", "key": 1, "value": "x"}',
        '{"type": "alias_matches", "aliases": "staging"}',
        "null",
    ],
)
def test_a_corrupt_stored_condition_denies(env, corrupted):
    make_model(env)
    grant(env, [(MODEL, "*", MANAGE.name), (MODEL, "*", READ.name, PROMOTED)])
    assert can_update("m")
    corrupt_conditions(env, corrupted)

    assert not can_read("m")
    assert not can_update("m")
    assert not can_delete("m")
    assert not can_set_alias("m", "staging")
    assert not can_delete_alias("m", "staging")
    assert visible_models(env) == []


def test_a_corrupt_condition_cannot_widen_what_another_role_denies(env):
    make_model(env)
    grant(env, [(MODEL, "*", DENY.name)])
    grant(env, [(MODEL, "*", MANAGE.name, PROMOTED)])
    corrupt_conditions(env, "not json")
    assert not can_read("m")


def test_a_corrupt_alias_condition_denies_alias_operations(env):
    make_model(env)
    grant(env, [(MODEL, "*", EDIT.name, ALIASES)])
    assert can_set_alias("m", "staging")
    corrupt_conditions(env, "{")
    assert not can_set_alias("m", "staging")
    assert not can_delete_alias("m", "staging")


def test_no_grants_is_still_denied(env):
    make_model(env)
    assert not can_read("m")
    assert not can_set_alias("m", "staging")
    assert visible_models(env) == []
