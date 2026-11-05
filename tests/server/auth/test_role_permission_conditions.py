from typing import NamedTuple

import pytest

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.server.auth.conditions import (
    CONDITION_ALIAS_MATCHES,
    CONDITION_TAG_EQUALS,
    CONDITION_TAG_NOT_EQUALS,
    UNSET,
    ConditionContext,
    deserialize_condition,
    evaluate_condition,
    grant_applies,
    serialize_condition,
    validate_condition,
)
from mlflow.server.auth.permissions import (
    DENY,
    EDIT,
    READ,
    RESOURCE_TYPE_EXPERIMENT,
    RESOURCE_TYPE_REGISTERED_MODEL,
    GrantLoadKey,
)
from mlflow.server.auth.requirements import applicable_grants, fold_grants_for_key

RM = RESOURCE_TYPE_REGISTERED_MODEL


class Grant(NamedTuple):
    resource_type: str
    resource_pattern: str
    permission: str
    condition: str | None = None


class LegacyGrant(NamedTuple):
    resource_type: str
    resource_pattern: str
    permission: str


def _tag_eq(key="promoted", value="true"):
    return {"type": CONDITION_TAG_EQUALS, "key": key, "value": value}


def _tag_ne(key="promoted", value="true"):
    return {"type": CONDITION_TAG_NOT_EQUALS, "key": key, "value": value}


def _alias(*aliases):
    return {"type": CONDITION_ALIAS_MATCHES, "aliases": list(aliases)}


def _raw(condition):
    return serialize_condition(condition)


def test_unset_repr():
    assert repr(UNSET) == "UNSET"


# --- validate_condition ---


def test_validate_none():
    assert validate_condition(None, RM) is None
    assert validate_condition(None, RESOURCE_TYPE_EXPERIMENT) is None


@pytest.mark.parametrize("condition", [_tag_eq(), _tag_ne("k", ""), _alias("a", "production-*")])
def test_validate_valid_returns_equal_copy(condition):
    result = validate_condition(condition, RM)
    assert result == condition
    assert result is not condition


def test_validate_copies_aliases_list():
    condition = _alias("a")
    result = validate_condition(condition, RM)
    result["aliases"].append("b")
    assert condition["aliases"] == ["a"]


@pytest.mark.parametrize(
    ("condition", "match"),
    [
        ("tag_equals", "must be a JSON object"),
        ([1], "must be a JSON object"),
        ({}, "Unknown condition type None"),
        ({"type": "foo"}, "Unknown condition type 'foo'. Supported: "),
        ({"type": 3}, "Unknown condition type 3"),
        ({"type": "tag_equals", "key": "k"}, "missing \\['value'\\]"),
        ({"type": "tag_equals", "key": "k", "value": "v", "x": 1}, "unexpected \\['x'\\]"),
        ({"type": "tag_equals", "key": "", "value": "v"}, "'key' must be a non-empty string"),
        ({"type": "tag_equals", "key": 1, "value": "v"}, "'key' must be a non-empty string"),
        ({"type": "tag_not_equals", "key": "k", "value": 1}, "'value' must be a string"),
        ({"type": "tag_not_equals", "key": "k", "value": None}, "'value' must be a string"),
        ({"type": "alias_matches"}, "missing \\['aliases'\\]"),
        ({"type": "alias_matches", "aliases": []}, "non-empty list"),
        ({"type": "alias_matches", "aliases": "prod"}, "non-empty list"),
        ({"type": "alias_matches", "aliases": ["a", ""]}, "non-empty list"),
        ({"type": "alias_matches", "aliases": ["a", 1]}, "non-empty list"),
        ({"type": "alias_matches", "aliases": ["a"], "key": "k"}, "unexpected \\['key'\\]"),
        ({"type": "alias_matches", "key": "k", "value": "v"}, "missing \\['aliases'\\]"),
    ],
)
def test_validate_rejects_malformed(condition, match):
    with pytest.raises(MlflowException, match=match) as exc:
        validate_condition(condition, RM)
    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_validate_rejects_non_registered_model_resource_type():
    with pytest.raises(MlflowException, match="only supported on 'registered_model'"):
        validate_condition(_tag_eq(), RESOURCE_TYPE_EXPERIMENT)


# --- serialisation ---


def test_serialize_is_canonical():
    a = {"type": CONDITION_TAG_EQUALS, "key": "k", "value": "v"}
    b = {"value": "v", "key": "k", "type": CONDITION_TAG_EQUALS}
    assert serialize_condition(a) == serialize_condition(b)
    assert serialize_condition(a) == '{"key":"k","type":"tag_equals","value":"v"}'


def test_serialize_none():
    assert serialize_condition(None) is None
    assert deserialize_condition(None) is None


@pytest.mark.parametrize("condition", [_tag_eq(), _tag_ne(), _alias("a", "b-*")])
def test_roundtrip(condition):
    assert deserialize_condition(serialize_condition(condition)) == condition


# --- evaluate_condition ---


def test_tag_equals():
    raw = _raw(_tag_eq())
    assert evaluate_condition(raw, ConditionContext(tags={"promoted": "true"})) is True
    assert evaluate_condition(raw, ConditionContext(tags={"promoted": "false"})) is False
    assert evaluate_condition(raw, ConditionContext(tags={})) is False


def test_tag_not_equals():
    raw = _raw(_tag_ne())
    assert evaluate_condition(raw, ConditionContext(tags={"promoted": "true"})) is False
    assert evaluate_condition(raw, ConditionContext(tags={"promoted": "false"})) is True
    assert evaluate_condition(raw, ConditionContext(tags={})) is True


@pytest.mark.parametrize("builder", [_tag_eq, _tag_ne])
def test_tag_condition_undecidable_without_tags(builder):
    raw = _raw(builder())
    assert evaluate_condition(raw, None) is None
    assert evaluate_condition(raw, ConditionContext()) is None
    assert evaluate_condition(raw, ConditionContext(alias="x")) is None


def test_tag_condition_ignores_alias():
    raw = _raw(_tag_eq())
    assert evaluate_condition(raw, ConditionContext(tags={"promoted": "true"}, alias="x")) is True


def test_alias_matches():
    raw = _raw(_alias("staging", "production-*"))
    assert evaluate_condition(raw, ConditionContext(alias="staging")) is True
    assert evaluate_condition(raw, ConditionContext(alias="production-eu")) is True
    assert evaluate_condition(raw, ConditionContext(alias="production")) is False
    assert evaluate_condition(raw, ConditionContext(alias="champion")) is False


def test_alias_matches_is_case_sensitive():
    raw = _raw(_alias("Staging", "Prod-*"))
    assert evaluate_condition(raw, ConditionContext(alias="staging")) is False
    assert evaluate_condition(raw, ConditionContext(alias="prod-1")) is False
    assert evaluate_condition(raw, ConditionContext(alias="Prod-1")) is True


def test_alias_matches_without_alias_is_false():
    raw = _raw(_alias("*"))
    assert evaluate_condition(raw, None) is False
    assert evaluate_condition(raw, ConditionContext(tags={"a": "b"})) is False


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "",
        "[]",
        "null",
        '"x"',
        "{}",
        '{"type":"foo"}',
        '{"type":"tag_equals"}',
        '{"type":"tag_equals","key":1,"value":"v"}',
        '{"type":"tag_equals","key":"k","value":2}',
        '{"type":"alias_matches"}',
        '{"type":"alias_matches","aliases":"x"}',
        '{"type":"alias_matches","aliases":[1]}',
    ],
)
def test_evaluate_malformed_is_none(raw):
    context = ConditionContext(tags={"k": "v"}, alias="x")
    assert evaluate_condition(raw, context) is None


def test_evaluate_never_raises_on_non_string_raw():
    assert evaluate_condition(None, ConditionContext()) is None
    assert evaluate_condition(5, ConditionContext()) is None


def test_evaluate_never_raises_on_bad_tags_object():
    raw = _raw(_tag_eq())
    assert evaluate_condition(raw, ConditionContext(tags=["not", "a", "mapping"])) is None


# --- grant_applies ---


def test_grant_applies_without_condition_attribute():
    assert grant_applies(LegacyGrant(RM, "*", EDIT.name), None) is True


def test_grant_applies_unconditional():
    assert grant_applies(Grant(RM, "*", EDIT.name), None) is True


def test_grant_applies_follows_evaluation():
    grant = Grant(RM, "*", EDIT.name, _raw(_tag_eq()))
    assert grant_applies(grant, ConditionContext(tags={"promoted": "true"})) is True
    assert grant_applies(grant, ConditionContext(tags={})) is False


def test_grant_applies_undecidable_fails_closed():
    positive = Grant(RM, "*", EDIT.name, _raw(_tag_eq()))
    deny = Grant(RM, "*", DENY.name, _raw(_tag_eq()))
    assert grant_applies(positive, None) is False
    assert grant_applies(deny, None) is True


def test_grant_applies_malformed_condition_fails_closed():
    assert grant_applies(Grant(RM, "*", EDIT.name, "garbage"), ConditionContext(tags={})) is False
    assert grant_applies(Grant(RM, "*", DENY.name, "garbage"), ConditionContext(tags={})) is True


# --- applicable_grants ---


def test_applicable_grants_keeps_unconditional_as_is():
    grants = [Grant(RM, "*", EDIT.name), LegacyGrant(RM, "m", READ.name)]
    result = applicable_grants(grants, None)
    assert result == grants
    assert result[0] is grants[0]
    assert result[1] is grants[1]


def test_applicable_grants_filters_and_clears_condition():
    edit = Grant(RM, "*", EDIT.name, _raw(_tag_ne()))
    read = Grant(RM, "*", READ.name, _raw(_tag_eq()))
    context = ConditionContext(tags={})
    assert applicable_grants([edit, read], context) == [Grant(RM, "*", EDIT.name)]
    context = ConditionContext(tags={"promoted": "true"})
    assert applicable_grants([edit, read], context) == [Grant(RM, "*", READ.name)]


def test_applicable_grants_undecidable():
    positive = Grant(RM, "*", EDIT.name, _raw(_tag_eq()))
    deny = Grant(RM, "*", DENY.name, _raw(_tag_eq()))
    assert applicable_grants([positive, deny], None) == [Grant(RM, "*", DENY.name)]


def test_applicable_grants_does_not_mutate_input():
    grants = [Grant(RM, "*", EDIT.name, _raw(_tag_ne()))]
    applicable_grants(grants, ConditionContext(tags={}))
    assert grants[0].condition is not None


def test_applicable_grants_empty():
    assert applicable_grants([], None) == []


def test_conditional_deny_blocks_only_when_condition_holds():
    key = GrantLoadKey(RM, "m")
    grants = [
        Grant(RM, "*", EDIT.name),
        Grant(RM, "*", DENY.name, _raw(_tag_eq())),
    ]
    promoted = ConditionContext(tags={"promoted": "true"})
    plain = ConditionContext(tags={})
    assert fold_grants_for_key(applicable_grants(grants, promoted), key) == DENY
    assert fold_grants_for_key(applicable_grants(grants, plain), key) == EDIT


def test_alias_condition_silent_without_alias():
    key = GrantLoadKey(RM, "m")
    grants = [Grant(RM, "*", EDIT.name, _raw(_alias("staging")))]
    assert fold_grants_for_key(applicable_grants(grants, ConditionContext(tags={})), key) is None
    assert (
        fold_grants_for_key(applicable_grants(grants, ConditionContext(alias="staging")), key)
        == EDIT
    )
    assert (
        fold_grants_for_key(applicable_grants(grants, ConditionContext(alias="champion")), key)
        is None
    )


# --- fold guard ---


def test_fold_guard_skips_unresolved_positive_conditional_grant():
    key = GrantLoadKey(RM, "m")
    assert fold_grants_for_key([Grant(RM, "*", EDIT.name, _raw(_tag_eq()))], key) is None


def test_fold_guard_unresolved_conditional_deny_still_denies():
    key = GrantLoadKey(RM, "m")
    assert fold_grants_for_key([Grant(RM, "*", DENY.name, _raw(_tag_eq()))], key) == DENY


def test_fold_guard_does_not_affect_other_grants():
    key = GrantLoadKey(RM, "m")
    grants = [
        Grant(RM, "*", EDIT.name, _raw(_tag_eq())),
        Grant(RM, "m", READ.name),
    ]
    assert fold_grants_for_key(grants, key) == READ


def test_fold_guard_ignores_non_matching_conditional_deny():
    key = GrantLoadKey(RM, "m")
    grants = [Grant(RM, "other", DENY.name, _raw(_tag_eq())), Grant(RM, "*", READ.name)]
    assert fold_grants_for_key(grants, key) == READ
