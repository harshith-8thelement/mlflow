import pytest

from mlflow.exceptions import MlflowException
from mlflow.server.auth.db.models import SqlPermissionAuditLog
from mlflow.server.auth.entities import PermissionAuditLogEntry
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore

from tests.helper_functions import random_str

pytestmark = pytest.mark.notrackingurimock


@pytest.fixture
def store(tmp_sqlite_uri):
    store = SqlAlchemyStore()
    store.init_db(tmp_sqlite_uri)
    return store


@pytest.fixture
def user(store):
    return store.create_user(random_str(), random_str())


@pytest.fixture
def user2(store):
    return store.create_user(random_str(), random_str())


def _add_rows(store, rows):
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add_all(SqlPermissionAuditLog(**row) for row in rows)


def test_grant_without_actor_writes_no_entry(store, user):
    store.grant_user_resource_permission(user.username, "experiment", "1", "READ")
    assert store.list_permission_audit_log() == []


def test_grant_with_actor_records_create_entry(store, user):
    store.grant_user_resource_permission(
        user.username, "experiment", "1", "READ", acting_username="admin"
    )
    (entry,) = store.list_permission_audit_log()
    assert isinstance(entry, PermissionAuditLogEntry)
    assert entry.action == "create"
    assert entry.actor_username == "admin"
    assert entry.target_username == user.username
    assert entry.resource_type == "experiment"
    assert entry.resource_id == "1"
    assert entry.old_permission is None
    assert entry.new_permission == "READ"
    assert entry.timestamp > 0


def test_grant_records_canonical_resource_id(store, user):
    store.grant_user_resource_permission(
        user.username, "experiment", "01", "READ", acting_username="admin"
    )
    (entry,) = store.list_permission_audit_log()
    assert entry.resource_id == "1"


def test_revoke_with_actor_records_delete_entry(store, user):
    store.grant_user_resource_permission(user.username, "experiment", "1", "EDIT")
    store.revoke_user_resource_permission(user.username, "experiment", "1", acting_username="admin")
    (entry,) = store.list_permission_audit_log()
    assert entry.action == "delete"
    assert entry.actor_username == "admin"
    assert entry.target_username == user.username
    assert entry.resource_type == "experiment"
    assert entry.resource_id == "1"
    assert entry.old_permission == "EDIT"
    assert entry.new_permission is None
    assert entry.timestamp > 0


def test_revoke_without_actor_writes_no_entry(store, user):
    store.grant_user_resource_permission(user.username, "experiment", "1", "EDIT")
    store.revoke_user_resource_permission(user.username, "experiment", "1")
    assert store.list_permission_audit_log() == []


def test_duplicate_grant_adds_no_entry(store, user):
    store.grant_user_resource_permission(
        user.username, "experiment", "1", "READ", acting_username="admin"
    )
    with pytest.raises(MlflowException, match="already exists") as exc:
        store.grant_user_resource_permission(
            user.username, "experiment", "1", "EDIT", acting_username="admin"
        )
    assert exc.value.error_code == "RESOURCE_ALREADY_EXISTS"
    assert len(store.list_permission_audit_log()) == 1


def test_grant_unknown_user_adds_no_entry(store):
    with pytest.raises(MlflowException, match="nobody"):
        store.grant_user_resource_permission(
            "nobody", "experiment", "1", "READ", acting_username="admin"
        )
    assert store.list_permission_audit_log() == []


def test_grant_invalid_resource_type_adds_no_entry(store, user):
    with pytest.raises(MlflowException, match="Invalid resource type"):
        store.grant_user_resource_permission(
            user.username, "bogus", "1", "READ", acting_username="admin"
        )
    assert store.list_permission_audit_log() == []


def test_grant_invalid_permission_adds_no_entry(store, user):
    with pytest.raises(MlflowException, match="ermission"):
        store.grant_user_resource_permission(
            user.username, "experiment", "1", "NOT_A_PERMISSION", acting_username="admin"
        )
    assert store.list_permission_audit_log() == []


def test_revoke_missing_grant_adds_no_entry(store, user):
    with pytest.raises(MlflowException, match="not found") as exc:
        store.revoke_user_resource_permission(
            user.username, "experiment", "1", acting_username="admin"
        )
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"
    assert store.list_permission_audit_log() == []


def test_revoke_unknown_user_adds_no_entry(store):
    with pytest.raises(MlflowException, match="nobody"):
        store.revoke_user_resource_permission(
            "nobody", "experiment", "1", acting_username="admin"
        )
    assert store.list_permission_audit_log() == []


def test_list_orders_newest_first(store, user):
    for exp_id in ("1", "2", "3"):
        store.grant_user_resource_permission(
            user.username, "experiment", exp_id, "READ", acting_username="admin"
        )
    entries = store.list_permission_audit_log()
    assert [e.resource_id for e in entries] == ["3", "2", "1"]


def test_list_orders_by_timestamp_then_id(store, user):
    _add_rows(
        store,
        [
            {
                "action": "create",
                "actor_username": "admin",
                "target_username": user.username,
                "resource_type": "experiment",
                "resource_id": resource_id,
                "new_permission": "READ",
                "timestamp": ts,
            }
            for resource_id, ts in (("a", 200), ("b", 100), ("c", 200))
        ],
    )
    assert [e.resource_id for e in store.list_permission_audit_log()] == ["c", "a", "b"]


def test_list_filters_by_username(store, user, user2):
    store.grant_user_resource_permission(
        user.username, "experiment", "1", "READ", acting_username="admin"
    )
    store.grant_user_resource_permission(
        user2.username, "experiment", "1", "READ", acting_username="admin"
    )
    entries = store.list_permission_audit_log(username=user.username)
    assert [e.target_username for e in entries] == [user.username]
    assert store.list_permission_audit_log(username="nobody") == []


def test_list_filters_by_resource_type(store, user):
    store.grant_user_resource_permission(
        user.username, "experiment", "1", "READ", acting_username="admin"
    )
    store.grant_user_resource_permission(
        user.username, "registered_model", "m", "READ", acting_username="admin"
    )
    entries = store.list_permission_audit_log(resource_type="registered_model")
    assert [(e.resource_type, e.resource_id) for e in entries] == [("registered_model", "m")]


def test_list_combines_filters(store, user, user2):
    for u in (user, user2):
        store.grant_user_resource_permission(
            u.username, "experiment", "1", "READ", acting_username="admin"
        )
        store.grant_user_resource_permission(
            u.username, "registered_model", "m", "READ", acting_username="admin"
        )
    entries = store.list_permission_audit_log(username=user2.username, resource_type="experiment")
    assert [(e.target_username, e.resource_type) for e in entries] == [
        (user2.username, "experiment")
    ]


def test_list_max_results_limits_to_newest(store, user):
    for exp_id in ("1", "2", "3"):
        store.grant_user_resource_permission(
            user.username, "experiment", exp_id, "READ", acting_username="admin"
        )
    assert [e.resource_id for e in store.list_permission_audit_log(max_results=2)] == ["3", "2"]


def test_list_default_max_results_is_100(store, user):
    _add_rows(
        store,
        [
            {
                "action": "create",
                "actor_username": "admin",
                "target_username": user.username,
                "resource_type": "experiment",
                "resource_id": str(i),
                "new_permission": "READ",
                "timestamp": i,
            }
            for i in range(120)
        ],
    )
    entries = store.list_permission_audit_log()
    assert len(entries) == 100
    assert entries[0].resource_id == "119"


def test_update_entry_round_trips(store, user):
    _add_rows(
        store,
        [
            {
                "action": "update",
                "actor_username": "admin",
                "target_username": user.username,
                "resource_type": "experiment",
                "resource_id": "1",
                "old_permission": "READ",
                "new_permission": "EDIT",
                "timestamp": 1,
            }
        ],
    )
    (entry,) = store.list_permission_audit_log()
    assert entry.action == "update"
    assert entry.old_permission == "READ"
    assert entry.new_permission == "EDIT"
    assert PermissionAuditLogEntry.from_json(entry.to_json()).to_json() == entry.to_json()
