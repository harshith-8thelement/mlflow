"""Conditions on role permissions: validation, serialisation and evaluation.

A condition narrows a role permission on a registered model to the cases where a tag or an
alias in the request matches. Everything here is pure -- the caller supplies the resource's
current attributes through :class:`ConditionContext` -- so the semantics are testable without
a store or a request.

Evaluation is three-valued: ``True`` (holds), ``False`` (does not hold) and ``None``
(cannot be decided). ``None`` is resolved fail-closed by :func:`grant_applies`: an
undecidable DENY applies, an undecidable positive grant does not.
"""

import json
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any, NamedTuple

from mlflow.exceptions import MlflowException
from mlflow.server.auth.permissions import DENY, RESOURCE_TYPE_REGISTERED_MODEL

if TYPE_CHECKING:
    from collections.abc import Mapping

CONDITION_TAG_EQUALS = "tag_equals"
CONDITION_TAG_NOT_EQUALS = "tag_not_equals"
CONDITION_ALIAS_MATCHES = "alias_matches"
CONDITION_TYPES = frozenset({
    CONDITION_TAG_EQUALS,
    CONDITION_TAG_NOT_EQUALS,
    CONDITION_ALIAS_MATCHES,
})

_TAG_CONDITION_TYPES = frozenset({CONDITION_TAG_EQUALS, CONDITION_TAG_NOT_EQUALS})


class _Unset:
    """Sentinel for "condition not provided" (an update then keeps the existing one)."""

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _Unset()


class ConditionContext(NamedTuple):
    # Persisted tags of the model; None means unknown (tag conditions are then undecidable).
    tags: "Mapping[str, str] | None" = None
    # Alias named by a set/delete-alias request; None for every other operation.
    alias: str | None = None


def _supported_types() -> str:
    return ", ".join(sorted(CONDITION_TYPES))


def validate_condition(condition: Any, resource_type: str) -> dict[str, Any] | None:
    """Validate ``condition`` and return a normalized copy (``None`` stays ``None``).

    Raises ``MlflowException`` (INVALID_PARAMETER_VALUE) naming the problem.
    """
    if condition is None:
        return None
    if resource_type != RESOURCE_TYPE_REGISTERED_MODEL:
        raise MlflowException.invalid_parameter_value(
            f"Conditions are only supported on '{RESOURCE_TYPE_REGISTERED_MODEL}' permissions, "
            f"not '{resource_type}'."
        )
    if not isinstance(condition, dict):
        raise MlflowException.invalid_parameter_value(
            f"Condition must be a JSON object, got {type(condition).__name__}."
        )
    condition_type = condition.get("type")
    if not isinstance(condition_type, str) or condition_type not in CONDITION_TYPES:
        raise MlflowException.invalid_parameter_value(
            f"Unknown condition type {condition_type!r}. Supported: {_supported_types()}."
        )

    if condition_type in _TAG_CONDITION_TYPES:
        expected = {"type", "key", "value"}
    else:
        expected = {"type", "aliases"}
    actual = set(condition)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        problems = []
        if missing:
            problems.append(f"missing {missing}")
        if extra:
            problems.append(f"unexpected {extra}")
        raise MlflowException.invalid_parameter_value(
            f"Invalid '{condition_type}' condition: {'; '.join(problems)}. "
            f"Expected exactly the fields {sorted(expected)}."
        )

    if condition_type in _TAG_CONDITION_TYPES:
        key, value = condition["key"], condition["value"]
        if not isinstance(key, str) or not key:
            raise MlflowException.invalid_parameter_value(
                f"Invalid '{condition_type}' condition: 'key' must be a non-empty string."
            )
        if not isinstance(value, str):
            raise MlflowException.invalid_parameter_value(
                f"Invalid '{condition_type}' condition: 'value' must be a string."
            )
        return {"type": condition_type, "key": key, "value": value}

    aliases = condition["aliases"]
    if (
        not isinstance(aliases, list)
        or not aliases
        or not all(isinstance(a, str) and a for a in aliases)
    ):
        raise MlflowException.invalid_parameter_value(
            f"Invalid '{condition_type}' condition: 'aliases' must be a non-empty list of "
            "non-empty strings."
        )
    return {"type": condition_type, "aliases": list(aliases)}


def serialize_condition(condition: dict[str, Any] | None) -> str | None:
    if condition is None:
        return None
    return json.dumps(condition, sort_keys=True, separators=(",", ":"))


def deserialize_condition(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    return json.loads(raw)


def evaluate_condition(raw: str, context: ConditionContext | None) -> bool | None:
    """Whether the serialized condition ``raw`` holds in ``context``; never raises.

    Returns ``None`` when it cannot be decided: ``raw`` is not a valid condition, or a tag
    condition has no tags to look at.
    """
    try:
        condition = json.loads(raw)
        if not isinstance(condition, dict):
            return None
        condition_type = condition.get("type")
        if condition_type in _TAG_CONDITION_TYPES:
            key, value = condition["key"], condition["value"]
            if not isinstance(key, str) or not isinstance(value, str):
                return None
            if context is None or context.tags is None:
                return None
            if condition_type == CONDITION_TAG_EQUALS:
                return context.tags.get(key) == value
            return context.tags.get(key) != value
        if condition_type == CONDITION_ALIAS_MATCHES:
            aliases = condition["aliases"]
            if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
                return None
            if context is None or context.alias is None:
                return False
            return any(fnmatchcase(context.alias, pattern) for pattern in aliases)
        return None
    except Exception:
        return None


def grant_applies(grant: Any, context: ConditionContext | None) -> bool:
    """Whether ``grant`` takes part in the fold for a request described by ``context``.

    An undecidable condition fails closed: a DENY applies, a positive grant does not.
    """
    raw = getattr(grant, "condition", None)
    if raw is None:
        return True
    result = evaluate_condition(raw, context)
    if result is None:
        return grant.permission == DENY.name
    return result
