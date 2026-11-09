"""Conditions on role permissions: validate, canonicalize, hash and evaluate.

A condition narrows a role permission so it applies only while the condition holds at
permission-check time. Everything here is pure -- evaluation takes a
:class:`ConditionContext` -- so it is testable without a store or a request.

Supported shapes (registered models only)::

    {"type": "tag_equals", "key": "<tag key>", "value": "<tag value>"}
    {"type": "tag_not_equals", "key": "<tag key>", "value": "<tag value>"}
    {"type": "alias_matches", "aliases": ["<alias or glob>", ...]}
"""

import fnmatch
import hashlib
import json
from collections.abc import Mapping
from typing import Any, NamedTuple

from mlflow.exceptions import MlflowException
from mlflow.server.auth.permissions import RESOURCE_TYPE_REGISTERED_MODEL

CONDITION_TYPE_TAG_EQUALS = "tag_equals"
CONDITION_TYPE_TAG_NOT_EQUALS = "tag_not_equals"
CONDITION_TYPE_ALIAS_MATCHES = "alias_matches"

CONDITION_RESOURCE_TYPES: frozenset[str] = frozenset({RESOURCE_TYPE_REGISTERED_MODEL})

_TAG_CONDITION_KEYS = frozenset({"type", "key", "value"})
_ALIAS_CONDITION_KEYS = frozenset({"type", "aliases"})
_CONDITION_TYPES = (
    CONDITION_TYPE_TAG_EQUALS,
    CONDITION_TYPE_TAG_NOT_EQUALS,
    CONDITION_TYPE_ALIAS_MATCHES,
)


class ConditionContext(NamedTuple):
    # None means there is no concrete resource, or it was not found.
    tags: "Mapping[str, str] | None"
    # Set only while checking an alias set/delete.
    alias: str | None


class ConditionEvaluationError(Exception):
    """A stored condition could not be evaluated. Callers must fail closed."""


def _check_exact_keys(condition: dict[str, Any], expected: frozenset[str], type_: str) -> None:
    missing = sorted(expected - condition.keys())
    if missing:
        raise MlflowException.invalid_parameter_value(
            f"Condition of type {type_!r} is missing required field(s): {missing}."
        )
    extra = sorted(str(k) for k in condition.keys() - expected)
    if extra:
        raise MlflowException.invalid_parameter_value(
            f"Condition of type {type_!r} has unexpected field(s): {extra}."
        )


def validate_condition(condition: object, resource_type: str) -> dict[str, Any]:
    """Return the canonical condition dict, or raise ``INVALID_PARAMETER_VALUE``."""
    if resource_type not in CONDITION_RESOURCE_TYPES:
        raise MlflowException.invalid_parameter_value(
            f"Conditions are not supported on resource type {resource_type!r}; "
            f"supported: {sorted(CONDITION_RESOURCE_TYPES)}."
        )
    if not isinstance(condition, dict):
        raise MlflowException.invalid_parameter_value(
            f"Condition must be an object, got {type(condition).__name__}."
        )
    if "type" not in condition:
        raise MlflowException.invalid_parameter_value("Condition is missing required field 'type'.")
    type_ = condition["type"]
    if not isinstance(type_, str) or type_ not in _CONDITION_TYPES:
        raise MlflowException.invalid_parameter_value(
            f"Unknown condition type {type_!r}; expected one of {list(_CONDITION_TYPES)}."
        )

    if type_ == CONDITION_TYPE_ALIAS_MATCHES:
        _check_exact_keys(condition, _ALIAS_CONDITION_KEYS, type_)
        aliases = condition["aliases"]
        if (
            not isinstance(aliases, list)
            or not aliases
            or not all(isinstance(a, str) and a for a in aliases)
        ):
            raise MlflowException.invalid_parameter_value(
                "Condition field 'aliases' must be a non-empty list of non-empty strings."
            )
        return {"type": type_, "aliases": list(aliases)}

    _check_exact_keys(condition, _TAG_CONDITION_KEYS, type_)
    key, value = condition["key"], condition["value"]
    if not isinstance(key, str) or not key:
        raise MlflowException.invalid_parameter_value(
            "Condition field 'key' must be a non-empty string."
        )
    if not isinstance(value, str):
        raise MlflowException.invalid_parameter_value("Condition field 'value' must be a string.")
    return {"type": type_, "key": key, "value": value}


def serialize_condition(condition: dict[str, Any] | None) -> str | None:
    if condition is None:
        return None
    return json.dumps(condition, sort_keys=True, separators=(",", ":"))


def condition_hash(serialized: str | None) -> str:
    """Stable identity of a serialized condition: ``""`` for none, else a SHA-256 hex digest."""
    if serialized is None:
        return ""
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def parse_condition(serialized: str | None) -> dict[str, Any] | None:
    if serialized is None:
        return None
    return json.loads(serialized)


def condition_holds(serialized: str, context: ConditionContext) -> bool:
    """Whether the serialized condition holds in ``context``.

    Raises :class:`ConditionEvaluationError` if the stored condition is malformed or of an
    unknown type; callers must treat that as a denial.
    """
    try:
        condition = json.loads(serialized)
    except (TypeError, ValueError) as e:
        raise ConditionEvaluationError(f"Malformed condition: {e}") from e
    if not isinstance(condition, dict):
        raise ConditionEvaluationError("Malformed condition: not an object")

    type_ = condition.get("type")
    try:
        if type_ == CONDITION_TYPE_TAG_EQUALS:
            return context.tags is not None and context.tags.get(condition["key"]) == (
                condition["value"]
            )
        if type_ == CONDITION_TYPE_TAG_NOT_EQUALS:
            return context.tags is not None and context.tags.get(condition["key"]) != (
                condition["value"]
            )
        if type_ == CONDITION_TYPE_ALIAS_MATCHES:
            if context.alias is None:
                return False
            return any(fnmatch.fnmatchcase(context.alias, p) for p in condition["aliases"])
    except (KeyError, TypeError) as e:
        raise ConditionEvaluationError(f"Malformed {type_!r} condition: {e!r}") from e
    raise ConditionEvaluationError(f"Unknown condition type {type_!r}")
