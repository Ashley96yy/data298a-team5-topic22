"""Basic input/output guardrails for the recommender (Models 0 and 1, 298A).

Scope (basic only; the deeper layer is planned for 298B):
  * validate_request():   reject malformed requests before they reach a model.
  * remove_seen():        never recommend an item the user already has.
  * sanitize_output():    drop invalid, out-of-catalog, seen or duplicate items from a model's list.
  * popularity_fallback(): fill or replace a list from training popularity.
  * guard_output():       sanitize, then fill from popularity if the list came back short or unusable.
  * guarded_recommend():  wraps any `recommend_fn(history, method, k) -> list[item_id]` with all of the above.
  * RejectionLog:         records ONLY a request id, a reason code and a time. Never the history, titles or text.

Item ids are 1..n_items; 0 is the padding index and is never a valid item. Pure Python + numpy, no extra dependencies.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import re
import uuid

import numpy as np

MAX_HISTORY = 50      # matches the frozen max sequence length; older items are dropped, not rejected
MAX_K = 50

# Reason codes. These are the only strings the rejection log may hold.
HISTORY_NOT_A_LIST = 'history_not_a_list'
HISTORY_ITEM_NOT_INTEGER = 'history_item_not_integer'
HISTORY_ITEM_OUT_OF_CATALOG = 'history_item_out_of_catalog'
INVALID_K = 'invalid_k'
UNKNOWN_METHOD = 'unknown_method'
EMPTY_HISTORY = 'empty_history'                  # flag, not a rejection: cold start goes to popularity
HISTORY_TRUNCATED = 'history_truncated'          # flag
OUTPUT_NOT_A_LIST = 'output_not_a_list'
OUTPUT_INVALID_ITEM = 'output_invalid_item'
OUTPUT_OUT_OF_CATALOG = 'output_out_of_catalog'
OUTPUT_SEEN_ITEM = 'output_seen_item'
OUTPUT_DUPLICATE = 'output_duplicate'
OUTPUT_TRUNCATED = 'output_truncated'
FALLBACK_FILL = 'fallback_fill'
RECOMMENDER_ERROR = 'recommender_error'

ALL_REASON_CODES = frozenset({
    HISTORY_NOT_A_LIST, HISTORY_ITEM_NOT_INTEGER, HISTORY_ITEM_OUT_OF_CATALOG, INVALID_K, UNKNOWN_METHOD,
    EMPTY_HISTORY, HISTORY_TRUNCATED, OUTPUT_NOT_A_LIST, OUTPUT_INVALID_ITEM, OUTPUT_OUT_OF_CATALOG,
    OUTPUT_SEEN_ITEM, OUTPUT_DUPLICATE, OUTPUT_TRUNCATED, FALLBACK_FILL, RECOMMENDER_ERROR,
})

_REQUEST_ID = re.compile(r'^[A-Za-z0-9_-]{1,64}$')


def _is_int(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _is_sequence(x):
    """Supported containers: list, tuple, or a one-dimensional numpy array.

    A zero-dimensional array has __iter__ but raises TypeError when iterated, so it must be checked by type and ndim.
    """
    if isinstance(x, np.ndarray):
        return x.ndim == 1
    return isinstance(x, (list, tuple))


def new_request_id():
    return uuid.uuid4().hex


def _safe_request_id(request_id):
    if request_id is None:
        return new_request_id()
    return request_id if isinstance(request_id, str) and _REQUEST_ID.match(request_id) else 'invalid-request-id'


@dataclass(frozen=True)
class RequestCheck:
    ok: bool
    request_id: str
    reason_code: str | None = None               # set when ok is False
    history: tuple = ()                          # what the model sees: the most recent MAX_HISTORY items, only when ok
    full_history: tuple = ()                     # every validated item; used for seen-item filtering and fallback
    k: int = 0
    flags: tuple = ()                            # non-fatal notes such as empty_history


def validate_request(history, k, n_items, request_id=None, method=None, methods=None):
    """Check one request. Returns a RequestCheck; never raises on bad user input."""
    rid = _safe_request_id(request_id)

    def reject(code):
        return RequestCheck(ok=False, request_id=rid, reason_code=code)

    if not _is_sequence(history):
        return reject(HISTORY_NOT_A_LIST)
    items = list(history)
    if not all(_is_int(i) for i in items):
        return reject(HISTORY_ITEM_NOT_INTEGER)
    items = [int(i) for i in items]
    if any(i < 1 or i > n_items for i in items):
        return reject(HISTORY_ITEM_OUT_OF_CATALOG)
    if not _is_int(k) or k < 1 or k > MAX_K:
        return reject(INVALID_K)
    if methods is not None and method not in methods:
        return reject(UNKNOWN_METHOD)
    flags = []
    model_items = items
    if len(items) > MAX_HISTORY:
        model_items = items[-MAX_HISTORY:]
        flags.append(HISTORY_TRUNCATED)
    if not items:
        flags.append(EMPTY_HISTORY)
    return RequestCheck(ok=True, request_id=rid, history=tuple(model_items), full_history=tuple(items),
                        k=int(k), flags=tuple(flags))


def remove_seen(candidates, history):
    """Candidates in their original order, without anything in the user's history."""
    seen = {int(i) for i in history}
    return [int(c) for c in candidates if int(c) not in seen]


def sanitize_output(item_ids, history, n_items, k):
    """Return (clean_ids, reason_codes). Keeps order, drops bad/seen/duplicate items, cuts to k."""
    if not _is_sequence(item_ids):
        return [], [OUTPUT_NOT_A_LIST]
    seen = {int(i) for i in history}
    clean, taken, codes = [], set(), set()
    for raw in item_ids:
        if not _is_int(raw):
            codes.add(OUTPUT_INVALID_ITEM)
            continue
        item = int(raw)
        if item < 1 or item > n_items:
            codes.add(OUTPUT_OUT_OF_CATALOG)
        elif item in seen:
            codes.add(OUTPUT_SEEN_ITEM)
        elif item in taken:
            codes.add(OUTPUT_DUPLICATE)
        else:
            clean.append(item)
            taken.add(item)
    if len(clean) > k:
        clean = clean[:k]
        codes.add(OUTPUT_TRUNCATED)
    return clean, sorted(codes)


def popularity_fallback(popularity, history, k, exclude=()):
    """Top-k most popular items (ties: lower id first), skipping padding, seen items and `exclude`.

    `popularity` is the training count array of length n_items + 1 (index 0 = padding).
    """
    pop = np.asarray(popularity, dtype=np.float64)
    blocked = {int(i) for i in history} | {int(i) for i in exclude}
    order = np.lexsort((np.arange(len(pop)), -pop))        # popularity desc, id asc
    out = []
    for i in order:
        if i == 0 or int(i) in blocked:
            continue
        out.append(int(i))
        if len(out) == k:
            break
    return out


@dataclass(frozen=True)
class OutputResult:
    items: list
    reason_codes: list
    used_fallback: bool


def guard_output(item_ids, history, popularity, k):
    """Sanitize a model's list; if it is short or unusable, fill the rest from popularity."""
    n_items = len(popularity) - 1
    clean, codes = sanitize_output(item_ids, history, n_items, k)
    used = False
    if len(clean) < k:
        clean = clean + popularity_fallback(popularity, history, k - len(clean), exclude=clean)
        codes = sorted(set(codes) | {FALLBACK_FILL})
        used = True
    return OutputResult(items=clean, reason_codes=codes, used_fallback=used)


class RejectionLog:
    """Keeps only {request_id, reason_code, time}. Nothing about the user or the request content."""

    def __init__(self):
        self.records = []

    def record(self, request_id, reason_code):
        if reason_code not in ALL_REASON_CODES:
            raise ValueError('unknown reason code')            # blocks free text from ever entering the log
        self.records.append({'request_id': _safe_request_id(request_id), 'reason_code': reason_code,
                             'time_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')})

    def counts(self):
        out = {}
        for r in self.records:
            out[r['reason_code']] = out.get(r['reason_code'], 0) + 1
        return out

    def write_jsonl(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            for r in self.records:
                f.write(json.dumps(r) + '\n')


def guarded_recommend(recommend_fn, popularity, history, k, method=None, methods=None, request_id=None, log=None):
    """Run one request through input check -> model -> output check -> popularity fallback.

    recommend_fn(history_list, method, k) -> list, tuple or 1-D array of item ids; it receives only the most recent
    MAX_HISTORY items, while the seen-item filter and the popularity fallback use the full validated history.
    Returns a dict with ok, request_id,
    items, reason_codes and used_fallback. A rejected request returns no items and one reason code.
    """
    n_items = len(popularity) - 1
    check = validate_request(history, k, n_items, request_id=request_id, method=method, methods=methods)
    if not check.ok:
        if log is not None:
            log.record(check.request_id, check.reason_code)
        return {'ok': False, 'request_id': check.request_id, 'items': [],
                'reason_codes': [check.reason_code], 'used_fallback': False}
    codes = list(check.flags)
    hist = list(check.full_history)                                 # seen-item filtering and fallback use everything
    try:
        raw = recommend_fn(list(check.history), method, check.k) if hist else []   # model: last MAX_HISTORY only
    except Exception:                                               # model failure must not break the request
        raw = []
        codes.append(RECOMMENDER_ERROR)
    out = guard_output(raw, hist, popularity, check.k)
    out_codes = set(out.reason_codes)
    if not hist:
        out_codes.discard(FALLBACK_FILL)                            # popularity for a cold start is expected
    codes = sorted(set(codes) | out_codes)
    if log is not None:
        for c in codes:
            if c not in (EMPTY_HISTORY, HISTORY_TRUNCATED):
                log.record(check.request_id, c)
    return {'ok': True, 'request_id': check.request_id, 'items': out.items,
            'reason_codes': codes, 'used_fallback': out.used_fallback}
