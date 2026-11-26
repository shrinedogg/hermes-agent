"""Pytest probe: track live pooled-connection checkouts across the session to find leaks.

Run: ``python -m pytest ... -p pgleak_probe`` from the repo root.

A single session-global counter tracks net checkouts (getconn - putconn) of the shared
per-DSN pool. After each test we report the running total and the delta for that test.
A total that keeps climbing test over test (never returning to baseline) is a true leak
(connections borrowed but never returned); a per-test spike that returns to baseline is
just handle churn.
"""
from __future__ import annotations

import traceback

import pytest

_state = {
    "live": 0,        # net checkouts right now
    "peak": 0,        # highest net checkouts seen
    "baseline": 0,    # net checkouts at session start
    "last_get_stack": "",
    "wrapped": False,
}


def _install():
    import persistence.postgres._backend as be

    if _state["wrapped"]:
        return
    orig = be._pool_for

    def pool_for(dsn):
        pool = orig(dsn)
        getconn, putconn = pool.getconn, pool.putconn
        if getattr(pool, "_leak_probe", False):
            return pool

        def w_getconn(*a, **k):
            conn = getconn(*a, **k)
            _state["live"] += 1
            if _state["live"] > _state["peak"]:
                _state["peak"] = _state["live"]
            _state["last_get_stack"] = "".join(traceback.format_stack()[-5:-1])
            return conn

        def w_putconn(conn, **k):
            _state["live"] -= 1
            return putconn(conn, **k)

        pool.getconn = w_getconn
        pool.putconn = w_putconn
        pool._leak_probe = True
        return pool

    be._pool_for = pool_for
    _state["wrapped"] = True
    _state["baseline"] = _state["live"]


@pytest.fixture(autouse=True)
def _leak_probe():
    _install()
    before = _state["live"]
    yield
    after = _state["live"]
    delta = after - before
    item = getattr(pytest, "current_test", None)
    name = item.name if item is not None else "?"
    if delta != 0 or after != _state["baseline"]:
        print(f"\n[pgleak] {name}: live={after} (baseline {_state['baseline']}, this-test delta {delta:+d})")
        if _state["live"] > 0 and _state["last_get_stack"]:
            print(_state["last_get_stack"])


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # no-op; keeps the hook table happy
    outcome = yield
    return outcome
