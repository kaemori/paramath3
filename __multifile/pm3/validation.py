from __future__ import annotations

from .codegen import generate_expression
from .constants import INFIX_OPERATIONS, OPERATION_EXPANSIONS, math_funcs
from .utils import num


def _is_number_like(token: str) -> bool:
    try:
        num(token)
        return True
    except Exception:
        return False


def _validate_balanced_parentheses(tokens: list, *, line_num: int | None = None):
    depth = 0
    for tok in tokens:
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
            if depth < 0:
                msg = "Unexpected closing parenthesis"
                if line_num is not None:
                    msg = f"Line {line_num}: {msg}"
                raise SyntaxError(msg) from None

    if depth != 0:
        msg = "Unexpected end of tokens"
        if line_num is not None:
            msg = f"Line {line_num}: {msg}"
        raise SyntaxError(msg) from None


def _validate_no_undefined_identifiers(
    ast,
    *,
    declared_variables: list[str] | None,
    line_num: int | None = None,
):
    allowed_vars = set(declared_variables or [])

    allowed_vars.add("ans")

    allowed_constants = {"epsilon", "pi", "euler"}
    allowed_ops = set(INFIX_OPERATIONS) | set(OPERATION_EXPANSIONS.keys())
    allowed_funcs = set(math_funcs.keys())

    def walk(node):
        if isinstance(node, list):
            if node and isinstance(node[0], str):
                head = node[0]
                if (
                    head.isidentifier()
                    and head not in allowed_ops
                    and head not in allowed_funcs
                ):
                    raise SyntaxError(
                        (
                            f"Line {line_num}: Malformed expression: {generate_expression(node)}"
                            if line_num is not None
                            else f"Malformed expression: {generate_expression(node)}"
                        )
                    ) from None
            for child in node[1:] if (node and isinstance(node[0], str)) else node:
                walk(child)
            return

        if isinstance(node, (int, float)):
            return

        if isinstance(node, str):
            if node in allowed_constants:
                return
            if _is_number_like(node):
                return

            if node in allowed_ops:
                return

            if node.isidentifier():
                if node not in allowed_vars:
                    msg = f"Undefined identifier: {node}"
                    if line_num is not None:
                        msg = f"Line {line_num}: {msg}"
                    raise SyntaxError(msg) from None
                return

            msg = f"Invalid token: {node!r}"
            if line_num is not None:
                msg = f"Line {line_num}: {msg}"
            raise SyntaxError(msg) from None

    walk(ast)


def _raise_if_conflicts_with_declared_variables(
    name: str,
    declared_variables: list | None,
    *,
    kind: str,
    line_num: int | None = None,
):
    declared_variables = declared_variables or []
    if name in declared_variables:
        msg = f"{kind} {name!r} conflicts with a name declared in //variables"
        if line_num is not None:
            msg = f"Line {line_num}: {msg}"
        raise ValueError(msg)
