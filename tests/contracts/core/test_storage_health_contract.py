"""Contract cases for ``persistence.core.health.StorageHealth``: the latch and the readiness probe answer without creating a store."""

from __future__ import annotations

import pytest

from persistence import SessionId, StorageHealth, StorageProbe, StorageState


def test_health_is_the_storage_health_protocol(backend, scope_a):
    assert isinstance(backend.open(scope_a).health, StorageHealth)


def test_probe_reports_a_missing_store_as_not_initialized_and_never_creates_it(backend, scope_a):
    store_a = backend.open(scope_a)
    assert store_a.health.probe() == StorageProbe(status="ok", detail="not initialized")
    assert store_a.health.probe() == StorageProbe(status="ok", detail="not initialized")
    assert store_a.health.storage_state() is StorageState.OK
    assert store_a.health.probe() == StorageProbe(status="ok", detail="not initialized")


def test_probe_reports_a_written_store_as_healthy(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    store_a.state_meta.set("probe-key", "value")
    assert store_a.health.probe() == StorageProbe(status="ok", detail=None)
    assert store_a.health.storage_state() is StorageState.OK
    assert store_b.health.probe() == StorageProbe(status="ok", detail="not initialized")


def test_reads_of_an_uninitialized_store_do_not_shrink_the_pool(backend, backend_name, scope_a):
    # CONTRACT: a read that finds an uncreated schema hands its checked-out connection back to the
    # pool. A bare close() on a pool connection leaks the pool's slot (the pool counts the
    # connection against its size until it is returned, and only reaps idle connections), so
    # enough uninitialized reads would eat the shared pool's 64 slots and turn every later
    # checkout into a 30-second timeout (the full-suite "couldn't get a connection" cascade).
    if backend_name != "postgres":
        pytest.skip("the shared connection pool is a postgres backend fact")
    from persistence.postgres._backend import _dsn, _pool_for

    pool = _pool_for(_dsn())  # the lazy process pool; created (and opened) on first use
    before = pool.get_stats()["pool_available"]
    for index in range(8):
        # One fresh read-only handle per read, never closed (the suites' handle pattern); the store
        # is never written, so every read takes the uncreated-schema discard path.
        store = backend.open(scope_a)
        assert store.sessions.get(SessionId(f"pool-probe-{index}")) is None
    after = pool.get_stats()["pool_available"]
    assert after >= before, (
        f"an uninitialized read must return its pool connection: available {before} -> {after}"
    )
