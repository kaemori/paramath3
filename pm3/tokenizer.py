from __future__ import annotations

from .constants import INFIX_OPERATIONS
from .logging import _preview_sequence, print_debug


def tokenize(line: str) -> list[str]:
    special_tokens = {"    ": "TAB", "\t": "TAB"}
    multi_char_ops = [key for key in INFIX_OPERATIONS if len(key) > 1]
    multi_char_ops += [":=", "//", "**"]

    for token, replacement in special_tokens.items():
        line = line.replace(token, f" __{replacement}__ ")

    for i, op in enumerate(multi_char_ops):
        line = line.replace(op, f" __MULTIOP_{i}__ ")

    splitters = ["(", ")", ",", "+", "-", "*", "/", "=", "!", "\n"]
    for splitter in splitters:
        line = line.replace(splitter, f" {splitter} ")
    for i, op in enumerate(multi_char_ops):
        line = line.replace(f"__MULTIOP_{i}__", op)

    while "  " in line:
        line = line.replace("  ", " ")

    tokens = line.strip().split(" ")
    print_debug(f"tokenize: {len(tokens)} tokens, head={_preview_sequence(tokens)}")
    return tokens
