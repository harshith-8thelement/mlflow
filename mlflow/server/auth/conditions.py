"""Conditions on role permissions: validation, serialisation and evaluation.

A condition narrows a role permission on registered models so that it applies only while the
condition holds for the resource's current attributes and the request's parameters. Everything
here is pure so the rules can be tested without a store or a request.

Accepted shapes::

    {"type": "tag_equals" | "tag_not_equals", "key": str, "value": str}
    {"type": "alias_matches", "aliases": [str, ...]}

Evaluation fails closed: a condition that cannot be evaluated raises
:class:`ConditionEvaluationError`, and callers treat that as a denial.
"""

import fnmatch
import hashlib
import json
from collections.abc import Mapping
from typing import Any, Final, NamedTuple

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.server.auth.permissions import RESOURCE_TYPE_REGISTERED_MODEL

CONDITION_TAG_EQUALS = "tag_equals"
CONDITION_TAG_NOT_EQUALS = "tag_not_equals"
CONDITION_ALIAS_MATCHES = "alias_matches"
CONDITION_TYPES = (CONDITION_ALIAS_MATCHES, CONDITION_TAG_EQUALS, CONDITION_TAG_NOT_EQUALS)

# Distinguishes "condition not supplied" from an explicit ``None`` (clear the condition) on update.
UNSET_CONDITION: Final = object()

_TAG_TYPES = (CONDITION_TAG_EQUALS, CONDITION_TAG_NOT_EQUALS)


class ConditionContext(NamedTuple):
    # None = the resource's attributes are unknown (e.g. the model does not exist yet).
    tags: "Mapping[str, str] | None" = None
    # None = the request is not an alias set/delete.
    alias: str | None = None


class ConditionEvaluationError(Exception):
    """A condition could not be evaluated; callers must deny."""


def _invalid(message: str) -> MlflowException:
    return MlflowException(f"Invalid condition: {message}", error_code=INVALID_PARAMETER_VALUE)


def validate_condition(condition: object, resource_type: str) -> "dict[str, Any] | None":
    """Validate ``condition`` for ``resource_type`` and return a normalised copy.

    ``None`` (no condition) is accepted for any resource type.
    """
    if condition is None:
        return None
    if resource_type != RESOURCE_TYPE_REGISTERED_MODEL:
        raise _invalid(
            f"conditions are only supported on {RESOURCE_TYPE_REGISTERED_MODEL!r} permissions, "
            f"not {resource_type!r}."
        )
    if not isinstance(condition, dict):
        raise _invalid(f"must be an object, got {type(condition).__name__}.")
    condition_type = condition.get("type")
    if condition_type not in CONDITION_TYPES:
        raise _invalid(
            f"'type' must be one of {', '.join(repr(t) for t in CONDITION_TYPES)}, "
            f"got {condition_type!r}."
        )
    if condition_type in _TAG_TYPES:
        allowed = {"type", "key", "value"}
        key = condition.get("key")
        value = condition.get("value")
        if not isinstance(key, str) or not key:
            raise _invalid(f"{condition_type!r} requires a non-empty string 'key'.")
        if not isinstance(value, str):
            raise _invalid(f"{condition_type!r} requires a string 'value'.")
    else:
        allowed = {"type", "aliases"}
        aliases = condition.get("aliases")
        if (
            not isinstance(aliases, list)
            or not aliases
            or not all(isinstance(a, str) and a for a in aliases)
        ):
            raise _invalid(f"{condition_type!r} requires a non-empty list of non-empty 'aliases'.")
    if extra := sorted(str(k) for k in condition if k not in allowed):
        raise _invalid(f"unexpected field(s) for {condition_type!r}: {', '.join(extra)}.")
    if condition_type in _TAG_TYPES:
        return {"type": condition_type, "key": key, "value": value}
    return {"type": condition_type, "aliases": list(aliases)}


def serialize_condition(condition: "dict | None") -> str | None:
    if condition is None:
        return None
    return json.dumps(condition, sort_keys=True, separators=(",", ":"))


def parse_condition(serialized: str | None) -> "dict | None":
    if serialized is None:
        return None
    try:
        parsed = json.loads(serialized)
    except (TypeError, ValueError) as e:
        raise ConditionEvaluationError(f"Condition is not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ConditionEvaluationError("Condition must be a JSON object.")
    return parsed


def condition_key(serialized: str | None) -> str:
    """A fixed-width identity for a serialised condition; ``""`` when there is none."""
    if serialized is None:
        return ""
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def is_alias_condition(condition: "str | dict | None") -> bool:
    try:
        parsed = parse_condition(condition) if isinstance(condition, str) else condition
    except ConditionEvaluationError:
        return False
    return isinstance(parsed, dict) and parsed.get("type") == CONDITION_ALIAS_MATCHES


def condition_holds(
    condition: "str | dict", context: "ConditionContext | None", *, is_deny: bool
) -> bool:
    """Whether ``condition`` holds for ``context``.

    ``is_deny`` selects the conservative answer when the resource's tags are unknown: a DENY
    row then applies, a positive row does not. Raises :class:`ConditionEvaluationError` for an
    unknown or malformed condition.
    """
    parsed = parse_condition(condition) if isinstance(condition, str) else condition
    if not isinstance(parsed, dict):
        raise ConditionEvaluationError("Condition must be an object.")
    condition_type = parsed.get("type")
    if condition_type in _TAG_TYPES:
        key, value = parsed.get("key"), parsed.get("value")
        if not isinstance(key, str) or not isinstance(value, str):
            raise ConditionEvaluationError(f"Malformed {condition_type!r} condition.")
        if context is None or context.tags is None:
            return is_deny
        actual = context.tags.get(key)
        return actual == value if condition_type == CONDITION_TAG_EQUALS else actual != value
    if condition_type == CONDITION_ALIAS_MATCHES:
        aliases = parsed.get("aliases")
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            raise ConditionEvaluationError(f"Malformed {condition_type!r} condition.")
        if context is None or context.alias is None:
            return False
        return any(fnmatch.fnmatchcase(context.alias, pattern) for pattern in aliases)
    raise ConditionEvaluationError(f"Unknown condition type {condition_type!r}.")


def tags_to_mapping(tags: object) -> dict[str, str]:
    """Normalise a tag container (mapping, or iterable of ``.key``/``.value`` objects)."""
    if tags is None:
        return {}
    if isinstance(tags, Mapping):
        return dict(tags)
    return {tag.key: tag.value for tag in tags}  # type: ignore[attr-defined]
