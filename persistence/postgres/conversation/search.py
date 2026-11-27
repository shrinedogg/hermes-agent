"""``SessionSearch`` on PostgreSQL over the store's rows: the contract protocol's query routes
(the module docstring of ``persistence.conversation.search`` spells them out: the sanitizer, the
word route with its substring and OR-relaxed retries, the ``tool`` role substring route and the
CJK routes) served by SQL reads of the store's schema, with the query routes, snippets and
context windows running in Python over the pool.

PostgreSQL keeps no derived search index: the stored rows are the index, so an admitted
``rebuild_search_index`` reports 0 rebuilt indexes, ``retry_deferred_search_recovery`` reports
False (nothing is ever deferred) and ``search_index_rebuild_status`` reads the deferred-rebuild
markers the recovery bookkeeping plants in ``state_meta`` (the ``fts_rebuild_high_water`` /
``fts_rebuild_progress`` keys, left to the caller). The candidate pool is one SQL join of
messages to sessions with the routes' shared filters pushed down; unfiltered searches therefore
read the store's rows. A tsvector + GIN index is the follow-up scale-up, not the contract floor.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Collection, Optional, Sequence

from persistence import MessageId, SessionId
from persistence.conversation.messages import CONTENT_JSON_PREFIX
from persistence.conversation.search import (
    FTS_TOOL_CONTENT_PREFIX_CHARS,
    MAX_SEARCH_QUERY_CHARS,
    SEARCH_RESULT_FIELDS,
    SearchContext,
    SearchHit,
    SearchIndexRebuildStatus,
    SearchSort,
)

from .._backend import PostgresStore, register_repository
from ._store.common import _phs

__all__ = ["PgSearchRepository"]

# The sources whose rows the substring (trigram-shaped) index skips, as the store's index does.
_TRIGRAM_EXCLUDED_SOURCES = ("cron", "subagent")
_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x3000, 0x303F), (0x3040, 0x309F),
               (0x30A0, 0x30FF), (0xAC00, 0xD7AF))
_OPERATORS = frozenset({"AND", "OR", "NOT"})
_SPECIAL_RE = re.compile(r"""[+{}():"^@/#&|~\[\]<>,;!?$=\\']""")
_QUOTED_RE = re.compile(r'"[^"]*"')
_UNIT_RE = re.compile(r'"[^"]*"|\S+')
_SNIPPET_TOKENS = 40


def _is_cjk(char: str) -> bool:
    return any(lo <= ord(char) <= hi for lo, hi in _CJK_RANGES)


def _count_cjk(text: str) -> int:
    return sum(1 for char in text if _is_cjk(char))


def _non_operator_tokens(text: str) -> list[str]:
    return [token for token in text.split() if token.upper() not in _OPERATORS]


def _quote_tokens(text: str) -> str:
    """Each non-operator word quoted, the operator words kept."""
    return " ".join(t if t.upper() in _OPERATORS else '"' + t.replace('"', '""') + '"' for t in text.split())


def _strip_cjk_wildcards(text: str) -> str:
    """Trailing ``*`` dropped from each non-operator word unless it is all stars."""
    if "*" not in text:
        return text
    return " ".join(t if t.upper() in _OPERATORS else (t.rstrip("*") or t) for t in text.split()) or text


def sanitize_query(query: str) -> str:
    """The store's query sanitizer: the cap, the quoted-phrase hold, the special-character strip,
    the wildcard and dangling-operator cleanups, the dotted/hyphenated/underscored phrase quoting."""
    query = query[:MAX_SEARCH_QUERY_CHARS]
    held: list[str] = []

    def _hold(match: re.Match[str]) -> str:
        held.append(match.group(0))
        return f"\x00Q{len(held) - 1}\x00"

    text = _QUOTED_RE.sub(_hold, query).replace('"', " ")
    text = _SPECIAL_RE.sub(" ", text)
    if "%" in text and not any(_is_cjk(char) for char in text):
        text = text.replace("%", " ")
    text = re.sub(r"\*+", "*", text)
    text = re.sub(r"(^|\s)\*", r"\1", text)
    text = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", text.strip())
    text = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", text.strip())
    text = re.sub(r"\b(\w+(?:[._-]\w+)+)\b", r'"\1"', text)
    for index, quoted in enumerate(held):
        text = text.replace(f"\x00Q{index}\x00", quoted)
    return text.strip()


def _or_relaxed(query: str) -> Optional[str]:
    """The implicit-AND query with its units ORed, or None for fewer than two units or an explicit OR / NOT."""
    units: list[str] = []
    for token in _UNIT_RE.findall(query):
        if token.upper() in {"OR", "NOT"}:
            return None
        if token.upper() != "AND":
            units.append(token)
    return " OR ".join(units) if len(units) >= 2 else None


def _word_char(char: str) -> bool:
    category = unicodedata.category(char)
    return category[0] in "LNM" or category == "Co"


def _fold_word(word: str) -> str:
    """One word as the word index stores it: combining marks dropped (also from characters that
    decompose into one), lowercased."""
    out = []
    for char in word:
        parts = unicodedata.normalize("NFD", char)
        if any(unicodedata.category(part) == "Mn" for part in parts):
            char = "".join(part for part in parts if unicodedata.category(part) != "Mn")
        out.append(char)
    return "".join(out).lower()


def words(text: str) -> list[tuple[int, int, str]]:
    """``(start, end, folded)`` for each word of ``text`` (maximal runs of letters, digits and marks)."""
    out: list[tuple[int, int, str]] = []
    start: Optional[int] = None
    for index, char in enumerate(text + " "):
        if index < len(text) and _word_char(char):
            start = index if start is None else start
        elif start is not None:
            out.append((start, index, _fold_word(text[start:index])))
            start = None
    return [item for item in out if item[2]]


def fold_char(char: str) -> str:
    """Case folding one character to one character, as the substring index does."""
    mapped = char.casefold()
    if len(mapped) != 1:
        mapped = char.lower()
    return mapped if len(mapped) == 1 else char


def fold_chars(text: str) -> str:
    return "".join(fold_char(char) for char in text)


def like_text(text: Optional[str]) -> str:
    """A stored text as the LIKE scans read it: up to its first NUL, empty for NULL."""
    return (text or "").split("\x00", 1)[0]


def _ascii_lower(value: str) -> str:
    return re.sub(r"[A-Z]+", lambda match: match.group().lower(), value)


def like_contains(text: Optional[str], term: str) -> bool:
    """SQL ``LIKE '%term%'``: ASCII case folding only."""
    return _ascii_lower(term) in _ascii_lower(like_text(text))


class QuerySyntaxError(ValueError):
    """Sanitized text the store's query grammar still rejects."""


@dataclass(frozen=True)
class _Phrase:
    text: str
    prefix: bool


@dataclass(frozen=True)
class _Node:
    op: str
    left: Any
    right: Any


def _bareword_char(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char in "_\x1a") or not char.isascii()


def _lex(text: str) -> list[Any]:
    """Phrases and operator words of a sanitized query; any other character is a syntax error."""
    out: list[Any] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char in " \t\n\r":
            i += 1
            continue
        if char == '"':
            end, parts = i + 1, []
            while True:
                close = text.find('"', end)
                if close < 0:
                    raise QuerySyntaxError("unterminated phrase")
                parts.append(text[end:close])
                if text.startswith('""', close):
                    parts.append('"')
                    end = close + 2
                    continue
                i = close + 1
                break
            phrase = "".join(parts)
        elif _bareword_char(char):
            start = i
            while i < len(text) and _bareword_char(text[i]):
                i += 1
            phrase = text[start:i]
            if phrase in _OPERATORS and not text.startswith("*", i):
                out.append(phrase)
                continue
        else:
            raise QuerySyntaxError(f"unexpected {char!r}")
        prefix = text.startswith("*", i)
        i += 1 if prefix else 0
        out.append(_Phrase(phrase, prefix))
    return out


def parse_query(text: str) -> Any:
    """The query tree: adjacent phrases bind tightest, then ``NOT``, ``AND``, ``OR``; all left-associative."""
    items = _lex(text)
    position = 0

    def level(operator: Optional[str]) -> Any:
        nonlocal position
        if operator is None:
            node = None
            while position < len(items) and isinstance(items[position], _Phrase):
                node = items[position] if node is None else _Node("AND", node, items[position])
                position += 1
            if node is None:
                raise QuerySyntaxError("operand expected")
            return node
        lower = {"OR": "AND", "AND": "NOT", "NOT": None}[operator]
        node = level(lower)
        while position < len(items) and items[position] == operator:
            position += 1
            node = _Node(operator, node, level(lower))
        return node

    if not items:
        raise QuerySyntaxError("empty query")
    tree = level("OR")
    if position != len(items):
        raise QuerySyntaxError("operator out of place")
    return tree


def _phrase_words(phrase: _Phrase) -> list[str]:
    return [word for _, _, word in words(phrase.text)]


def _word_hits(phrase: _Phrase, text: str) -> list[tuple[int, int]]:
    """Character spans of ``phrase`` in ``text`` under word matching (the last word as a prefix with ``*``)."""
    needle = _phrase_words(phrase)
    found = words(text)
    spans: list[tuple[int, int]] = []
    for i in range(len(found) - len(needle) + 1 if needle else 0):
        run = [word for _, _, word in found[i : i + len(needle)]]
        last_ok = run[-1].startswith(needle[-1]) if phrase.prefix else run[-1] == needle[-1]
        if run[:-1] == needle[:-1] and last_ok:
            spans.append((found[i][0], found[i + len(needle) - 1][1]))
    return spans


def _substring_hits(phrase: _Phrase, text: str) -> list[tuple[int, int]]:
    """Spans of ``phrase`` in ``text`` as a case-folded substring; a phrase under three characters matches without a span."""
    if len(phrase.text) < 3:
        return [(0, 0)]
    needle, haystack = fold_chars(phrase.text), fold_chars(text)
    return [(m.start(), m.end()) for m in re.finditer(re.escape(needle), haystack)]


def matched_phrases(tree: Any, fields: Sequence[str], hits: Any) -> list[_Phrase]:
    """The phrases of the branches of ``tree`` a row with searchable columns ``fields`` matches
    (empty when the row does not match); a phrase matches inside one column."""
    if isinstance(tree, _Phrase):
        return [tree] if any(hits(tree, text) for text in fields) else []
    left = matched_phrases(tree.left, fields, hits)
    if tree.op == "NOT":
        return left if left and not matched_phrases(tree.right, fields, hits) else []
    right = matched_phrases(tree.right, fields, hits)
    if tree.op == "AND":
        return left + right if left and right else []
    return left + right


def _marked(text: str, spans: Sequence[tuple[int, int]], start: int, end: int) -> str:
    """``text[start:end]`` with every span inside it (overlaps merged) between ``>>>`` and ``<<<``;
    the store's snippet writer drops plain text from a NUL to the next match."""
    merged: list[list[int]] = []
    for lo, hi in sorted(span for span in spans if span[0] < span[1] and span[0] >= start and span[1] <= end):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    parts, cursor = [], start
    for lo, hi in merged:
        parts += [text[cursor:lo].split("\x00", 1)[0], ">>>", text[lo:hi], "<<<"]
        cursor = hi
    parts.append(text[cursor:end].split("\x00", 1)[0])
    return "".join(parts)


def word_snippet(columns: Sequence[str], phrases: Sequence[_Phrase]) -> str:
    """An excerpt of the first column holding a match of ``phrases``: up to 40 words around the first
    one, matches marked, ``...`` where cut."""
    for text in columns:
        spans = [span for phrase in phrases for span in _word_hits(phrase, text)]
        if not spans:
            continue
        found = words(text)
        if len(found) <= _SNIPPET_TOKENS:
            return _marked(text, spans, 0, len(text))
        first = next(i for i, item in enumerate(found) if item[0] == min(spans)[0])
        lo = min(max(0, first - (_SNIPPET_TOKENS - 1) // 2), len(found) - _SNIPPET_TOKENS)
        hi = lo + _SNIPPET_TOKENS
        start = found[lo][0] if lo else 0
        end = found[hi - 1][1] if hi < len(found) else len(text)
        return ("..." if lo else "") + _marked(text, spans, start, end) + ("..." if hi < len(found) else "")
    return columns[0] if columns else ""


def substring_snippet(columns: Sequence[str], phrases: Sequence[_Phrase]) -> str:
    """The first column holding a substring match of ``phrases``, matches marked; long text cut to
    about 40 characters around the first."""
    for text in columns:
        spans = [span for phrase in phrases for span in _substring_hits(phrase, text) if span[1] > span[0]]
        if not spans:
            continue
        if len(text) <= _SNIPPET_TOKENS + 2:
            return _marked(text, spans, 0, len(text))
        lo = max(0, min(min(spans)[0] - _SNIPPET_TOKENS // 2, len(text) - _SNIPPET_TOKENS))
        hi = lo + _SNIPPET_TOKENS
        return ("..." if lo else "") + _marked(text, spans, lo, hi) + ("..." if hi < len(text) else "")
    return columns[0] if columns else ""


def like_snippet(content: Optional[str], term: str) -> Optional[str]:
    """``substr(content, max(1, instr(content, term) - 40), 120)``: unmarked, from the start when
    the term does not occur case-sensitively."""
    if content is None:
        return None
    start = max(1, content.find(term) + 1 - 40)
    return content[start - 1 : start - 1 + 120]


def like_boolean(query: str) -> tuple[list[list[tuple[str, bool]]], Optional[str]]:
    """The ``tool`` role route's predicate: OR groups of (word, negated), groups without a positive
    word dropped, and the snippet word (``_compile_like_boolean_query``)."""
    groups: list[list[tuple[str, bool]]] = [[]]
    negate = False
    for token in _UNIT_RE.findall(query):
        operator = token.upper()
        if operator == "OR":
            if groups[-1]:
                groups.append([])
            negate = False
        elif operator == "NOT":
            negate = True
        elif operator not in {"AND", "NEAR"}:
            term = token.strip('"').strip("*").strip()
            if term:
                groups[-1].append((term, negate))
                negate = False
    kept = [group for group in groups if any(not negated for _, negated in group)]
    first = next((term for group in kept for term, negated in group if not negated), None)
    return kept, first


def _search_fields(fields: Optional[Collection[str]]) -> Optional[tuple[str, ...]]:
    """The requested projection in canonical order (``_search_message_fields``)."""
    if fields is None:
        return None
    if isinstance(fields, str):
        raise TypeError("search fields must be a collection of field names, not a string")
    requested = set(fields)
    unknown = requested.difference(SEARCH_RESULT_FIELDS)
    if unknown:
        raise ValueError(f"unknown search result field(s): {', '.join(sorted(unknown))}")
    return tuple(field for field in SEARCH_RESULT_FIELDS if field in requested)


def _page(rows: list[Any], limit: int, offset: int) -> list[Any]:
    """SQL ``LIMIT ? OFFSET ?``: a negative limit is no limit, a negative offset is zero."""
    start = max(0, offset)
    return rows[start:] if limit < 0 else rows[start : start + limit]


def _ranked(rows: list["_Row"], sort: SearchSort) -> list["_Row"]:
    """Word and substring route order: timestamp for NEWEST / OLDEST, equal rank (id order)
    otherwise and within a timestamp."""
    if sort is SearchSort.NEWEST:
        return sorted(rows, key=lambda r: (-r.timestamp, r.id))
    if sort is SearchSort.OLDEST:
        return sorted(rows, key=lambda r: (r.timestamp, r.id))
    return sorted(rows, key=lambda r: r.id)


def _scanned(rows: list["_Row"], sort: SearchSort) -> list["_Row"]:
    """``tool`` role route order: (timestamp, id) ascending for OLDEST, descending otherwise."""
    return sorted(rows, key=lambda r: (r.timestamp, r.id), reverse=sort is not SearchSort.OLDEST)


@dataclass
class _Session:
    session_id: str
    source: Optional[str]
    model: Optional[str]
    started_at: Optional[float]
    delegate_child: bool = False


@dataclass
class _Row:
    id: int
    session: _Session
    role: str
    content: Optional[str]
    timestamp: float
    tool_name: Optional[str]
    tool_calls: Optional[str]

    def word_columns(self) -> list[str]:
        content = (self.content or "")[:FTS_TOOL_CONTENT_PREFIX_CHARS] if self.role == "tool" else self.content or ""
        return [content, self.tool_name or "", self.tool_calls or ""]

    def substring_indexed(self) -> bool:
        return self.role != "tool" and self.session.source not in _TRIGRAM_EXCLUDED_SOURCES and not self.session.delegate_child

    def substring_columns(self) -> list[str]:
        return [self.content or "", self.tool_name or ""]

    def like_columns(self) -> list[Optional[str]]:
        return [self.content, self.tool_name, self.tool_calls]


def _decode_payload(payload: Any) -> Optional[str]:
    """A stored content BYTEA payload as its stored text (a str passthrough, the ``\\x00json:``
    encoding kept as stored); None stays None."""
    if payload is None or isinstance(payload, str):
        return payload
    return bytes(payload).decode("utf-8")


def _is_delegate_child(model_config: Optional[str]) -> bool:
    """The ``_delegate_from`` creation marker of the session's stored ``model_config``."""
    if not model_config:
        return False
    try:
        value = json.loads(model_config)
    except (TypeError, ValueError):
        return False
    return isinstance(value, dict) and "_delegate_from" in value


def _flatten_content(stored: Optional[str]) -> str:
    """A stored content payload as the context window reads it: multimodal parts flattened to
    joined text (a parts list without text as the placeholder), a string passed through, anything
    else empty (``flatten_search_text``)."""
    if stored is None:
        return ""
    if stored.startswith(CONTENT_JSON_PREFIX):
        raw = stored[len(CONTENT_JSON_PREFIX):]
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            value = raw
        if isinstance(value, list):
            parts = [p.get("text", "") for p in value if isinstance(p, dict) and p.get("type") == "text"]
            return " ".join(t for t in parts if t).strip() or "[multimodal content]"
        return value if isinstance(value, str) else ""
    return stored


def _word_route(pool: list[_Row], tree: Any, sort: SearchSort, limit: int, offset: int) -> list[tuple[_Row, str]]:
    phrases = {row.id: matched_phrases(tree, row.word_columns(), _word_hits) for row in pool}
    matched = [row for row in pool if phrases[row.id]]
    return [(row, word_snippet(row.word_columns(), phrases[row.id])) for row in _page(_ranked(matched, sort), limit, offset)]


def _substring_route(pool: list[_Row], tree: Any, sort: SearchSort, limit: int, offset: int) -> list[tuple[_Row, str]]:
    phrases = {row.id: matched_phrases(tree, row.substring_columns(), _substring_hits) for row in pool if row.substring_indexed()}
    matched = [row for row in pool if phrases.get(row.id)]
    return [(row, substring_snippet(row.substring_columns(), phrases[row.id])) for row in _page(_ranked(matched, sort), limit, offset)]


def _substring_retry(pool: list[_Row], text: str, sort: SearchSort, limit: int, offset: int) -> list[tuple[_Row, str]]:
    """The zero-result retry over the substring index, for a query whose words all have three or more characters."""
    bare = text.strip('"').strip()
    tokens = _non_operator_tokens(bare)
    if not tokens or any(len(token) < 3 for token in tokens):
        return []
    try:
        return _substring_route(pool, parse_query(_quote_tokens(bare)), sort, limit, offset)
    except QuerySyntaxError:
        return []


def _tool_route(pool: list[_Row], text: str, sort: SearchSort, limit: int, offset: int) -> list[tuple[_Row, Optional[str]]]:
    groups, first = like_boolean(text)
    if first is None:
        return []

    def present(row: _Row, term: str) -> bool:
        return any(like_contains(column, term) for column in row.like_columns())

    matched = [row for row in pool if any(all(present(row, term) != negated for term, negated in group) for group in groups)]
    return [(row, like_snippet(row.content, first)) for row in _page(_scanned(matched, sort), limit, offset)]


def _cjk_route(pool: list[_Row], text: str, unindexed: bool, sort: SearchSort, limit: int, offset: int) -> list[tuple[_Row, Optional[str]]]:
    """CJK text: the substring index when every CJK word has three or more CJK characters, else any
    word by LIKE, newest first."""
    raw = _strip_cjk_wildcards(text).strip('"').strip()
    tokens = _non_operator_tokens(raw)
    short = any(_count_cjk(token) < 3 for token in tokens if any(_is_cjk(char) for char in token))
    if _count_cjk(raw) >= 3 and not short and not unindexed:
        try:
            return _substring_route(pool, parse_query(_quote_tokens(raw)), sort, limit, offset)
        except QuerySyntaxError:
            pass
    terms = tokens or [raw]
    matched = [row for row in pool if any(like_contains(column, term) for term in terms for column in row.like_columns())]
    ordered = sorted(matched, key=lambda r: (r.timestamp, r.id), reverse=True)
    return [(row, like_snippet(row.content, terms[0])) for row in _page(ordered, limit, offset)]


def _context(row: _Row, by_session: dict[str, list[tuple[int, str, Optional[str], float]]]) -> tuple[SearchContext, ...]:
    """The previous row, the row and the next row of its session by (timestamp, id), whatever
    their visibility."""
    rows = by_session.get(row.session.session_id, [])
    if not any(item[0] == row.id for item in rows):
        return ()
    index = next(i for i, item in enumerate(rows) if item[0] == row.id)
    return tuple(SearchContext(item[1], _flatten_content(item[2])[:200])
                 for item in rows[max(0, index - 1) : index + 2])


def _hit(row: _Row, snippet: Optional[str], by_session: dict[str, list[tuple[int, str, Optional[str], float]]],
         projection: Optional[tuple[str, ...]]) -> SearchHit:
    values: dict[str, Any] = {
        "id": MessageId(row.id), "session_id": SessionId(row.session.session_id), "role": row.role, "snippet": snippet,
        "timestamp": row.timestamp, "tool_name": row.tool_name, "source": row.session.source, "model": row.session.model,
        "session_started": row.session.started_at,
    }
    wanted = SEARCH_RESULT_FIELDS if projection is None else projection
    if "context" in wanted:
        values["context"] = _context(row, by_session)
    names = {"id": "message_id"}
    return SearchHit(**{names.get(name, name): values[name] for name in wanted})


class PgSearchRepository:
    """Message search and index recovery over the bound store's schema."""

    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    # ── index administration ─────────────────────────────────────────────────

    def rebuild_search_index(self) -> int:
        """PostgreSQL keeps no derived search index (the stored rows are the index), so an admitted
        rebuild reports 0 rebuilt indexes and leaves any pending deferred rebuild untouched."""
        return 0

    def retry_deferred_search_recovery(self) -> bool:
        """Nothing is ever deferred on PostgreSQL; the verb reports False and never raises."""
        return False

    def search_index_rebuild_status(self) -> Optional[SearchIndexRebuildStatus]:
        """The deferred-rebuild progress from the ``state_meta`` markers, or None when nothing is
        pending (a non-positive high water)."""
        high_water = self._store.scalar(
            "SELECT value FROM state_meta WHERE key = %s", "fts_rebuild_high_water")
        if high_water is None or int(high_water) <= 0:
            return None
        progress = self._store.scalar(
            "SELECT value FROM state_meta WHERE key = %s", "fts_rebuild_progress") or "0"
        return SearchIndexRebuildStatus(total=int(high_water), indexed=int(progress))

    # ── search ────────────────────────────────────────────────────────────────

    def _pool(self, include_inactive: bool, source_filter: Optional[Sequence[str]], exclude_sources: Optional[Sequence[str]],
              role_filter: Optional[Sequence[str]], after_ts: Optional[int], before_ts: Optional[int]) -> list[_Row]:
        """The candidate rows: one join of messages to sessions with the routes' shared filters."""
        where: list[str] = []
        params: list[Any] = []
        if source_filter is not None and not source_filter:
            return []  # CONTRACT: an empty source filter keeps none
        if not include_inactive:
            where.append("(m.active = 1 OR m.compacted = 1)")
        where.append("COALESCE(m.display_kind, '') <> 'hidden'")
        if source_filter is not None:
            where.append(f"s.source IN ({_phs(len(source_filter))})")
            params.extend(source_filter)
        if exclude_sources:
            where.append(f"s.source IS NULL OR s.source NOT IN ({_phs(len(exclude_sources))})")
            params.extend(exclude_sources)
        if role_filter:
            where.append(f"m.role IN ({_phs(len(role_filter))})")
            params.extend(role_filter)
        if after_ts is not None:
            where.append("s.started_at >= %s")
            params.append(int(after_ts))
        if before_ts is not None:
            where.append("s.started_at < %s")
            params.append(int(before_ts))
        rows = self._store.all(
            "SELECT m.id, m.session_id, m.role, m.content, m.tool_name, m.tool_calls, m.timestamp, "
            "s.source, s.model, s.started_at, s.model_config "
            f"FROM messages m JOIN sessions s ON s.id = m.session_id WHERE {' AND '.join(where)}",
            *params)
        return [_Row(
            id=int(row["id"]),
            session=_Session(str(row["session_id"]), row["source"], row["model"], row["started_at"],
                             _is_delegate_child(row["model_config"])),
            role=row["role"], content=_decode_payload(row["content"]), timestamp=float(row["timestamp"]),
            tool_name=row["tool_name"], tool_calls=row["tool_calls"]) for row in rows]

    def _context_rows(self, session_ids: Sequence[str]) -> dict[str, list[tuple[int, str, Optional[str], float]]]:
        """Every stored row of the named sessions (whatever its visibility), (timestamp, id) ordered."""
        if not session_ids:
            return {}
        rows = self._store.all(
            "SELECT session_id, id, role, content, timestamp FROM messages "
            "WHERE session_id = ANY(%s) ORDER BY timestamp, id", list(session_ids))
        out: dict[str, list[tuple[int, str, Optional[str], float]]] = {}
        for row in rows:
            out.setdefault(str(row["session_id"]), []).append(
                (int(row["id"]), row["role"], _decode_payload(row["content"]), float(row["timestamp"])))
        return out

    def search_messages(
        self, query: str, *, source_filter: Optional[Sequence[str]] = None, exclude_sources: Optional[Sequence[str]] = None,
        role_filter: Optional[Sequence[str]] = None, limit: int = 20, offset: int = 0,
        sort: SearchSort = SearchSort.RELEVANCE, include_inactive: bool = False,
        fields: Optional[Collection[str]] = None, after_ts: Optional[int] = None,
        before_ts: Optional[int] = None) -> tuple[SearchHit, ...]:
        projection = _search_fields(fields)
        if not query or not query.strip():
            return ()
        text = sanitize_query(query)
        if not text:
            return ()
        pool = self._pool(include_inactive, source_filter, exclude_sources, role_filter, after_ts, before_ts)
        tool_route = bool(role_filter) and "tool" in (role_filter or ())
        if tool_route:
            found = _tool_route(pool, text, sort, limit, offset)
        elif any(_is_cjk(char) for char in text):
            unindexed = bool(source_filter) and any(src in _TRIGRAM_EXCLUDED_SOURCES for src in source_filter or ())
            found = _cjk_route(pool, text, unindexed, sort, limit, offset)
        else:
            try:
                found = _word_route(pool, parse_query(text), sort, limit, offset)
            except QuerySyntaxError:
                return ()
            if not found:
                found = _substring_retry(pool, text, sort, limit, offset)
            relaxed = _or_relaxed(text) if not found else None
            if relaxed is not None:
                try:
                    found = _word_route(pool, parse_query(relaxed), sort, limit, offset)
                except QuerySyntaxError:
                    found = []
        by_session = self._context_rows({row.session.session_id for row, _ in found}) if found and (
            projection is None or "context" in projection) else {}
        return tuple(_hit(row, snippet, by_session, projection) for row, snippet in found)


def _register() -> None:
    register_repository("search", lambda store: PgSearchRepository(store),
                        {"full_text_search": True, "substring_search": True})


_register()
