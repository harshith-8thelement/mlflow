from typing import NamedTuple

import pytest

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.server.auth.conditions import (
    CONDITION_ALIAS_MATCHES,
    CONDITION_TAG_EQUALS,
    CONDITION_TAG_NOT_EQUALS,
    CONDITION_TYPES,
    UNSET_CONDITION,
    ConditionContext,
    ConditionEvaluationError,
    condition_holds,
    condition_key,
    is_alias_condition,
    parse_condition,
    serialize_condition,
    tags_to_mapping,
    validate_condition,
)
from mlflow.server.auth.permissions import (
    DENY,
    EDIT,
    MANAGE,
    READ,
    RESOURCE_TYPE_EXPERIMENT,
    RESOURCE_TYPE_REGISTERED_MODEL,
    GrantLoadKey,
)
from mlflow.server.auth.requirements import alias_scoped_permission, fold_grants_for_key


class Row(NamedTuple):
    resource_type: str
    resource_pattern: str
    permission: str
    condition: str | None = None


MODEL = RESOURCE_TYPE_REGISTERED_MODEL
KEY = GrantLoadKey(MODEL, "m")
NOT_PROMOTED = {"type": CONDITION_TAG_NOT_EQUALS, "key": "promoted", "value": "true"}
PROMOTED = {"type": CONDITION_TAG_EQUALS, "key": "promoted", "value": "true"}
ALIASES = {"type": CONDITION_ALIAS_MATCHES, "aliases": ["staging", "production-*"]}


def row(permission, condition=None, pattern="*"):
    return Row(MODEL, pattern, permission.name, serialize_condition(condition))


# ------------------------------------------------------------- validation


@pytest.mark.parametrize("condition", [NOT_PROMOTED, PROMOTED, ALIASES])
def test_validate_accepts_each_shape_and_returns_a_copy(condition):
    result = validate_condition(condition, MODEL)
    assert result == condition
    assert result is not condition


def test_validate_none_is_none_for_any_resource_type():
    assert validate_condition(None, MODEL) is None
    assert validate_condition(None, RESOURCE_TYPE_EXPERIMENT) is None


def test_validate_accepts_empty_tag_value():
    cond = {"type": CONDITION_TAG_EQUALS, "key": "k", "value": ""}
    assert validate_condition(cond, MODEL) == cond


@pytest.mark.parametrize(
    ("condition", "resource_type", "message"),
    [
        (PROMOTED, RESOURCE_TYPE_EXPERIMENT, "only supported on 'registered_model'"),
        ("tag_equals", MODEL, "must be an object"),
        ([PROMOTED], MODEL, "must be an object"),
        ({}, MODEL, "'type' must be one of"),
        ({"type": "owner_is", "key": "a", "value": "b"}, MODEL, "'type' must be one of"),
        ({"type": CONDITION_TAG_EQUALS, "value": "v"}, MODEL, "non-empty string 'key'"),
        ({"type": CONDITION_TAG_EQUALS, "key": "", "value": "v"}, MODEL, "non-empty string 'key'"),
        ({"type": CONDITION_TAG_EQUALS, "key": 1, "value": "v"}, MODEL, "non-empty string 'key'"),
        ({"type": CONDITION_TAG_EQUALS, "key": "k"}, MODEL, "string 'value'"),
        ({"type": CONDITION_TAG_NOT_EQUALS, "key": "k", "value": 1}, MODEL, "string 'value'"),
        ({"type": CONDITION_ALIAS_MATCHES}, MODEL, "non-empty list"),
        ({"type": CONDITION_ALIAS_MATCHES, "aliases": []}, MODEL, "non-empty list"),
        ({"type": CONDITION_ALIAS_MATCHES, "aliases": "prod"}, MODEL, "non-empty list"),
        ({"type": CONDITION_ALIAS_MATCHES, "aliases": ["ok", ""]}, MODEL, "non-empty list"),
        ({"type": CONDITION_ALIAS_MATCHES, "aliases": ["ok", 1]}, MODEL, "non-empty list"),
        ({**PROMOTED, "extra": 1}, MODEL, r"unexpected field\(s\)"),
        ({**ALIASES, "key": "k"}, MODEL, r"unexpected field\(s\)"),
    ],
)
def test_validate_rejects_with_a_clear_error(condition, resource_type, message):
    with pytest.raises(MlflowException, match=message) as exc:
        validate_condition(condition, resource_type)
    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_validate_message_lists_supported_types():
    with pytest.raises(MlflowException, match="alias_matches.*tag_equals.*tag_not_equals"):
        validate_condition({"type": "nope"}, MODEL)


def test_condition_types_and_sentinel():
    assert set(CONDITION_TYPES) == {"tag_equals", "tag_not_equals", "alias_matches"}
    assert UNSET_CONDITION is not None
    assert UNSET_CONDITION is UNSET_CONDITION


# ------------------------------------------------- serialisation and key


def test_serialize_is_canonical_and_round_trips():
    a = serialize_condition({"value": "true", "key": "promoted", "type": CONDITION_TAG_EQUALS})
    b = serialize_condition(PROMOTED)
    assert a == b == '{"key":"promoted","type":"tag_equals","value":"true"}'
    assert parse_condition(a) == PROMOTED


def test_none_round_trips():
    assert serialize_condition(None) is None
    assert parse_condition(None) is None


@pytest.mark.parametrize("bad", ["not json", "[1]", '"s"', "5", "null"])
def test_parse_rejects_bad_json_and_non_objects(bad):
    with pytest.raises(ConditionEvaluationError, match="Condition"):
        parse_condition(bad)


def test_condition_key():
    assert condition_key(None) == ""
    k1 = condition_key(serialize_condition(PROMOTED))
    k2 = condition_key(serialize_condition(NOT_PROMOTED))
    assert len(k1) == 64
    assert k1 != k2
    assert k1 == condition_key(serialize_condition(PROMOTED))


def test_is_alias_condition():
    assert is_alias_condition(ALIASES)
    assert is_alias_condition(serialize_condition(ALIASES))
    assert not is_alias_condition(PROMOTED)
    assert not is_alias_condition(None)
    assert not is_alias_condition("not json")
    assert not is_alias_condition("[]")


# ------------------------------------------------------------- evaluation


@pytest.mark.parametrize(
    ("condition", "tags", "expected"),
    [
        (PROMOTED, {"promoted": "true"}, True),
        (PROMOTED, {"promoted": "false"}, False),
        (PROMOTED, {}, False),
        (NOT_PROMOTED, {"promoted": "true"}, False),
        (NOT_PROMOTED, {"promoted": "false"}, True),
        (NOT_PROMOTED, {}, True),
    ],
)
def test_tag_conditions_follow_current_tags(condition, tags, expected):
    ctx = ConditionContext(tags=tags)
    assert condition_holds(condition, ctx, is_deny=False) is expected
    assert condition_holds(serialize_condition(condition), ctx, is_deny=False) is expected


@pytest.mark.parametrize("condition", [PROMOTED, NOT_PROMOTED])
@pytest.mark.parametrize("context", [None, ConditionContext(), ConditionContext(alias="a")])
def test_unknown_tags_apply_only_to_deny(condition, context):
    assert condition_holds(condition, context, is_deny=True) is True
    assert condition_holds(condition, context, is_deny=False) is False


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("staging", True),
        ("production-eu", True),
        ("production-", True),
        ("production", False),
        ("Staging", False),
        ("champion", False),
        ("pre-production-eu", False),
    ],
)
def test_alias_matches_uses_case_sensitive_globs(alias, expected):
    ctx = ConditionContext(tags={}, alias=alias)
    assert condition_holds(ALIASES, ctx, is_deny=False) is expected


@pytest.mark.parametrize("context", [None, ConditionContext(), ConditionContext(tags={"a": "b"})])
@pytest.mark.parametrize("is_deny", [True, False])
def test_alias_condition_never_holds_without_an_alias(context, is_deny):
    assert condition_holds(ALIASES, context, is_deny=is_deny) is False


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "owner_is"},
        {},
        {"type": CONDITION_TAG_EQUALS, "key": "k"},
        {"type": CONDITION_TAG_EQUALS, "key": 1, "value": "v"},
        {"type": CONDITION_ALIAS_MATCHES, "aliases": "x"},
        {"type": CONDITION_ALIAS_MATCHES, "aliases": [1]},
        "garbage",
        "[1]",
    ],
)
def test_malformed_condition_raises(bad):
    with pytest.raises(ConditionEvaluationError, match=r"."):
        condition_holds(bad, ConditionContext(tags={}, alias="a"), is_deny=False)


def test_tags_to_mapping():
    class T(NamedTuple):
        key: str
        value: str

    assert tags_to_mapping(None) == {}
    assert tags_to_mapping({"a": "1"}) == {"a": "1"}
    assert tags_to_mapping([T("a", "1"), T("b", "2")]) == {"a": "1", "b": "2"}
    assert tags_to_mapping(iter([T("a", "1")])) == {"a": "1"}


# ------------------------------------------------------- the fold (tags)

TAG_ROWS = [row(EDIT, NOT_PROMOTED), row(READ, PROMOTED)]


def test_tag_conditions_switch_the_granted_level_as_tags_change():
    untagged = ConditionContext(tags={})
    promoted = ConditionContext(tags={"promoted": "true"})
    assert fold_grants_for_key(TAG_ROWS, KEY, untagged) == EDIT
    assert fold_grants_for_key(TAG_ROWS, KEY, promoted) == READ
    assert fold_grants_for_key(TAG_ROWS, KEY, ConditionContext(tags={})) == EDIT


def test_unknown_tags_make_positive_conditional_rows_silent():
    assert fold_grants_for_key(TAG_ROWS, KEY, None) is None
    assert fold_grants_for_key(TAG_ROWS, KEY, ConditionContext()) is None


def test_unknown_tags_apply_a_conditional_deny():
    rows = [row(MANAGE), row(DENY, PROMOTED)]
    assert fold_grants_for_key(rows, KEY, None) == DENY
    assert fold_grants_for_key(rows, KEY, ConditionContext(tags={})) == MANAGE
    assert fold_grants_for_key(rows, KEY, ConditionContext(tags={"promoted": "true"})) == DENY


def test_rows_without_a_condition_ignore_the_context():
    rows = [
        Row(MODEL, "*", READ.name),
        Row(MODEL, "m", MANAGE.name, None),
        Row(MODEL, "m", EDIT.name, ""),
    ]
    assert fold_grants_for_key(rows, KEY) == MANAGE
    assert fold_grants_for_key(rows, KEY, ConditionContext(tags={"x": "y"})) == MANAGE


def test_rows_without_a_condition_field_still_fold():
    class Legacy(NamedTuple):
        resource_type: str
        resource_pattern: str
        permission: str

    assert fold_grants_for_key([Legacy(MODEL, "*", EDIT.name)], KEY) == EDIT


def test_conditional_rows_still_respect_resource_matching():
    rows = [row(EDIT, NOT_PROMOTED, pattern="other"), row(READ, NOT_PROMOTED, pattern="m")]
    assert fold_grants_for_key(rows, KEY, ConditionContext(tags={})) == READ


@pytest.mark.parametrize("bad", ["not json", '{"type":"owner_is"}', '{"type":"tag_equals"}', "[1]"])
def test_unevaluable_condition_denies(bad):
    rows = [Row(MODEL, "*", MANAGE.name), Row(MODEL, "*", EDIT.name, bad)]
    assert fold_grants_for_key(rows, KEY, ConditionContext(tags={})) == DENY


def test_unevaluable_condition_denies_even_without_context():
    rows = [Row(MODEL, "*", EDIT.name, "not json")]
    assert fold_grants_for_key(rows, KEY) == DENY


# ------------------------------------------------------- the fold (alias)


def test_alias_rows_grant_nothing_to_non_alias_actions():
    rows = [row(EDIT, ALIASES)]
    assert fold_grants_for_key(rows, KEY, ConditionContext(tags={})) is None
    assert fold_grants_for_key(rows, KEY, None) is None


def test_alias_rows_grant_only_for_matching_aliases_in_the_fold():
    rows = [row(EDIT, ALIASES)]
    assert fold_grants_for_key(rows, KEY, ConditionContext(tags={}, alias="staging")) == EDIT
    assert fold_grants_for_key(rows, KEY, ConditionContext(tags={}, alias="champion")) is None


def test_alias_scoped_permission_only_considers_alias_rows():
    rows = [row(MANAGE), row(EDIT, NOT_PROMOTED), row(READ, ALIASES)]
    ctx = ConditionContext(tags={}, alias="production-eu")
    assert alias_scoped_permission(rows, KEY, ctx) == READ


def test_alias_scoped_permission_none_when_no_alias_row_applies():
    rows = [row(MANAGE), row(EDIT, ALIASES, pattern="other")]
    ctx = ConditionContext(alias="staging")
    assert alias_scoped_permission(rows, KEY, ctx) is None
    champion = ConditionContext(alias="champion")
    assert alias_scoped_permission([row(EDIT, ALIASES)], KEY, champion) is None


@pytest.mark.parametrize("context", [None, ConditionContext(), ConditionContext(tags={})])
def test_alias_scoped_permission_none_without_an_alias(context):
    assert alias_scoped_permission([row(EDIT, ALIASES)], KEY, context) is None


def test_alias_scoped_permission_takes_the_highest_and_deny_wins():
    ctx = ConditionContext(alias="staging")
    other = {"type": CONDITION_ALIAS_MATCHES, "aliases": ["stag*"]}
    assert alias_scoped_permission([row(READ, ALIASES), row(EDIT, other)], KEY, ctx) == EDIT
    rows = [row(MANAGE, ALIASES), row(DENY, other)]
    assert alias_scoped_permission(rows, KEY, ctx) == DENY


def test_alias_scoped_permission_denies_on_unreadable_condition():
    rows = [row(EDIT, ALIASES), Row(MODEL, "*", EDIT.name, "not json")]
    assert alias_scoped_permission(rows, KEY, ConditionContext(alias="staging")) == DENY


def test_alias_scoped_permission_ignores_other_well_formed_types():
    rows = [row(EDIT, PROMOTED)]
    ctx = ConditionContext(tags={"promoted": "true"}, alias="a")
    assert alias_scoped_permission(rows, KEY, ctx) is None
