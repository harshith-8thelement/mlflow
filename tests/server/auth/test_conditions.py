import json

import pytest

from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.server.auth.conditions import (
    CONDITION_RESOURCE_TYPES,
    ConditionContext,
    ConditionEvaluationError,
    condition_hash,
    condition_holds,
    parse_condition,
    serialize_condition,
    validate_condition,
)
from mlflow.server.auth.permissions import (
    RESOURCE_TYPE_EXPERIMENT,
    RESOURCE_TYPE_REGISTERED_MODEL,
)

MODEL = RESOURCE_TYPE_REGISTERED_MODEL


def _ctx(tags=None, alias=None):
    return ConditionContext(tags=tags, alias=alias)


def _holds(condition, context):
    return condition_holds(serialize_condition(condition), context)


def test_only_registered_models_support_conditions():
    assert CONDITION_RESOURCE_TYPES == frozenset({MODEL})


@pytest.mark.parametrize(
    "condition",
    [
        {"type": "tag_equals", "key": "promoted", "value": "true"},
        {"type": "tag_not_equals", "key": "promoted", "value": ""},
        {"type": "alias_matches", "aliases": ["champion", "production-*"]},
    ],
)
def test_validate_returns_canonical_dict(condition):
    assert validate_condition(condition, MODEL) == condition


def test_validate_returns_a_copy():
    aliases = ["a"]
    result = validate_condition({"type": "alias_matches", "aliases": aliases}, MODEL)
    assert result["aliases"] is not aliases


@pytest.mark.parametrize(
    ("condition", "resource_type", "match"),
    [
        ({"type": "tag_equals", "key": "k", "value": "v"}, RESOURCE_TYPE_EXPERIMENT, "resource"),
        (None, MODEL, "object"),
        ("tag_equals", MODEL, "object"),
        ([], MODEL, "object"),
        ({}, MODEL, "'type'"),
        ({"type": "regex", "key": "k", "value": "v"}, MODEL, "Unknown condition type"),
        ({"type": 1}, MODEL, "Unknown condition type"),
        ({"type": "tag_equals", "value": "v"}, MODEL, "'key'"),
        ({"type": "tag_equals", "key": "k"}, MODEL, "'value'"),
        ({"type": "tag_equals", "key": "k", "value": "v", "x": 1}, MODEL, "unexpected"),
        ({"type": "tag_equals", "key": "", "value": "v"}, MODEL, "'key'"),
        ({"type": "tag_equals", "key": 1, "value": "v"}, MODEL, "'key'"),
        ({"type": "tag_not_equals", "key": "k", "value": 1}, MODEL, "'value'"),
        ({"type": "tag_not_equals", "key": "k", "value": None}, MODEL, "'value'"),
        ({"type": "alias_matches"}, MODEL, "'aliases'"),
        ({"type": "alias_matches", "aliases": []}, MODEL, "'aliases'"),
        ({"type": "alias_matches", "aliases": "prod"}, MODEL, "'aliases'"),
        ({"type": "alias_matches", "aliases": [""]}, MODEL, "'aliases'"),
        ({"type": "alias_matches", "aliases": ["a", 1]}, MODEL, "'aliases'"),
        ({"type": "alias_matches", "aliases": ["a"], "key": "k"}, MODEL, "unexpected"),
    ],
)
def test_validate_rejects_bad_conditions(condition, resource_type, match):
    with pytest.raises(MlflowException, match=match) as exc:
        validate_condition(condition, resource_type)
    assert exc.value.error_code == databricks_pb2.ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_serialization_is_canonical():
    a = serialize_condition({"type": "tag_equals", "key": "k", "value": "v"})
    b = serialize_condition({"value": "v", "key": "k", "type": "tag_equals"})
    assert a == b == '{"key":"k","type":"tag_equals","value":"v"}'


def test_none_round_trips():
    assert serialize_condition(None) is None
    assert parse_condition(None) is None
    assert condition_hash(None) == ""


def test_parse_inverts_serialize():
    condition = {"type": "alias_matches", "aliases": ["a", "b-*"]}
    assert parse_condition(serialize_condition(condition)) == condition


def test_hash_is_stable_and_distinguishes_conditions():
    one = serialize_condition({"type": "tag_equals", "key": "k", "value": "v"})
    two = serialize_condition({"type": "tag_equals", "key": "k", "value": "w"})
    assert len(condition_hash(one)) == 64
    assert condition_hash(one) == condition_hash(one)
    assert condition_hash(one) != condition_hash(two)
    assert condition_hash(one) != ""


def test_tag_equals():
    c = {"type": "tag_equals", "key": "promoted", "value": "true"}
    assert _holds(c, _ctx({"promoted": "true"}))
    assert not _holds(c, _ctx({"promoted": "false"}))
    assert not _holds(c, _ctx({}))
    assert not _holds(c, _ctx(None))


def test_tag_not_equals():
    c = {"type": "tag_not_equals", "key": "promoted", "value": "true"}
    assert not _holds(c, _ctx({"promoted": "true"}))
    assert _holds(c, _ctx({"promoted": "false"}))
    assert _holds(c, _ctx({}))
    assert not _holds(c, _ctx(None))


def test_tag_conditions_follow_the_current_tags():
    c = {"type": "tag_not_equals", "key": "promoted", "value": "true"}
    assert _holds(c, _ctx({}))
    assert not _holds(c, _ctx({"promoted": "true"}))
    assert _holds(c, _ctx({}))


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("champion", True),
        ("production-1", True),
        ("production-", True),
        ("production", False),
        ("staging", False),
        ("Champion", False),
        (None, False),
    ],
)
def test_alias_matches(alias, expected):
    c = {"type": "alias_matches", "aliases": ["champion", "production-*"]}
    assert _holds(c, _ctx({"x": "y"}, alias)) is expected


def test_alias_matches_does_not_need_tags():
    c = {"type": "alias_matches", "aliases": ["a"]}
    assert _holds(c, _ctx(None, "a"))


@pytest.mark.parametrize(
    "serialized",
    [
        "not json",
        "[]",
        "null",
        json.dumps({"type": "nope"}),
        json.dumps({}),
        json.dumps({"type": "tag_equals"}),
        json.dumps({"type": "alias_matches", "aliases": 5}),
    ],
)
def test_condition_holds_raises_on_malformed_conditions(serialized):
    with pytest.raises(ConditionEvaluationError, match=r"."):
        condition_holds(serialized, _ctx({"k": "v"}, "a"))
