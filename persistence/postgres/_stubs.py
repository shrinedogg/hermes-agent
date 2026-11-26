"""D8: every verb a PostgreSQL repository has not ported yet raises ``NotImplementedError`` naming it.

A repository module declares the verbs it ports; :func:`add_stubs` then adds a stub for every other
callable member of the protocol, so the class carries every protocol attribute (the suites' protocol
``isinstance`` checks) and each unported verb fails loudly with its own name.
"""

from __future__ import annotations

from typing import Any

__all__ = ["add_stubs"]


def add_stubs(cls: type, protocol: type, implemented: tuple[str, ...] = ()) -> type:
    """Add a raising stub for every protocol member of ``protocol`` that ``cls`` does not already carry.

    ``implemented`` is the module's ported verb list (redundant with ``hasattr`` but keeps the D8
    inventory explicit); the stub raises ``NotImplementedError`` naming the verb.
    """
    for member in sorted(getattr(protocol, "__protocol_attrs__", ()) or ()):
        if member.startswith("_") or member in implemented or hasattr(cls, member):
            continue

        setattr(cls, member, _make_stub(member))
    return cls


def _make_stub(member: str):
    """A raising stub that names ``member`` in its failure message (factory: the message must bind the
    verb per member, not the loop's final value)."""
    def _stub(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"postgres {type(self).__name__}.{member} not ported (D8)")

    _stub.__name__ = member
    return _stub
