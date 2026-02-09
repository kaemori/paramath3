from __future__ import annotations

from .constants import INFIX_OPERATIONS, precedence, right_assoc
from .logging import _preview_sequence, print_debug


def generate_ast(tokens: list[str]):
    tokens = [t for t in tokens if t and t != "\n"]

    print_debug(f"generate_ast: {len(tokens)} tokens, head={_preview_sequence(tokens)}")

    def parse_expr(idx):
        result = []
        i = idx

        while i < len(tokens):
            token = tokens[i]

            if token == ")":
                return result, i
            elif token == ",":
                return result, i
            elif (
                i + 1 < len(tokens)
                and tokens[i + 1] == "("
                and isinstance(token, str)
                and token.isidentifier()
            ):
                func_name = token
                args = []
                i += 2

                while i < len(tokens) and tokens[i] != ")":
                    if tokens[i] == ",":
                        i += 1
                        continue
                    arg, i = parse_expr(i)
                    args.append(arg if len(arg) != 1 else arg[0])

                result.append([func_name] + args)
                i += 1
            elif token == "(":
                i += 1
                inner, i = parse_expr(i)
                result.append(inner if len(inner) != 1 else inner[0])
                i += 1
            else:
                result.append(token)
                i += 1

        return result, i

    ast, end_idx = parse_expr(0)
    if end_idx != len(tokens):
        bad = tokens[end_idx]
        print_debug(
            f"generate_ast: stopped early at token {end_idx}/{len(tokens)}: {bad!r}"
        )
        if bad == ")":
            raise SyntaxError("Unexpected closing parenthesis")
        if bad == ",":
            raise SyntaxError("Unexpected comma")
        raise SyntaxError(f"Unexpected token: {bad!r}")
    ast = ast if len(ast) != 1 else ast[0]
    print_debug(f"generate_ast: produced ast type={type(ast).__name__}")
    return ast


def infix_to_postfix(ast):
    def is_op(token):
        return isinstance(token, str) and token in INFIX_OPERATIONS

    def convert(node):
        if not isinstance(node, list):
            return node

        if not node:
            return node

        if len(node) > 1 and isinstance(node[1], str) and node[1] in INFIX_OPERATIONS:
            items = node

            print_debug(
                "infix_to_postfix: parsing infix list, "
                + f"items={_preview_sequence(items, max_items=24)}"
            )

            i = 0

            def peek():
                return items[i] if i < len(items) else None

            def consume():
                nonlocal i
                tok = items[i]
                i += 1
                return tok

            def parse_atom():
                tok = peek()
                if tok in ("+", "-"):
                    op = consume()
                    rhs = parse_atom()
                    if op == "+":
                        return rhs
                    return ["-", 0, rhs]
                if tok == "!":
                    consume()
                    rhs = parse_atom()
                    return ["!", rhs]

                tok = consume()
                if isinstance(tok, list):
                    return convert(tok)
                return tok

            def parse_expr(min_prec=0):
                left = parse_atom()
                while True:
                    op = peek()
                    if not is_op(op) or op not in precedence:
                        break
                    prec = precedence[op]
                    if prec < min_prec:
                        break

                    consume()
                    next_min = prec if op in right_assoc else prec + 1
                    right = parse_expr(next_min)
                    left = [op, left, right]

                return left

            result = parse_expr(0)
            if i != len(items):
                leftover = " ".join(str(x) for x in items[i:])
                print_debug(
                    f"infix_to_postfix: leftover tokens after parse: {leftover!r}"
                )
                raise SyntaxError(
                    f"Could not fully parse expression, leftover tokens: {leftover}"
                )
            return result

        if isinstance(node[0], str):
            return [node[0]] + [convert(arg) for arg in node[1:]]

        return [convert(elem) for elem in node]

    return convert(ast)
