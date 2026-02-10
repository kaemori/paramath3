from __future__ import annotations


DEBUG = False
VERBOSE = False
LOGFILE = None


def _log_message(prefix: str, message: str, max_len: int = 127):
    original_len = len(message)
    truncated = message[:max_len]
    log_line = f"[{prefix} {original_len}] {truncated}{'...' if original_len > max_len else ''}"
    print(log_line)
    if LOGFILE:
        with open(LOGFILE, "a", encoding="utf-8") as f:
            f.write(f"[{prefix} {original_len}] {message}\n")


def print_debug(msg: str):
    if DEBUG:
        _log_message("DEBUG", msg)


def print_verbose(msg: str):
    if VERBOSE:
        _log_message("VERBOSE", msg)


def _preview_sequence(seq, *, max_items: int = 16) -> str:
    try:
        n = len(seq)
    except Exception:
        return repr(seq)

    if n <= max_items:
        return repr(seq)
    head = list(seq[:max_items])
    return f"{head!r}... (+{n - max_items} more)"
