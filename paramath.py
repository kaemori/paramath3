import math
import json
import builtins
import re
from dataclasses import dataclass, replace
from typing import Union, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed


DEBUG = False
VERBOSE = False
LOGFILE = None
OPERATION_EXPANSIONS = {
    "==": lambda a, b, eps: [
        "/",
        eps,
        ["+", ["abs", ["-", a, b]], eps],
    ],
    "!=": lambda a, b, eps: [
        "/",
        ["abs", ["-", a, b]],
        ["+", ["abs", ["-", a, b]], eps],
    ],
    "=0": lambda a, eps: [
        "/",
        eps,
        ["+", ["abs", a], eps],
    ],
    ">": lambda a, b, eps: [
        "/",
        ["+", ["-", a, b], ["abs", ["-", a, b]]],
        ["+", ["*", 2, ["abs", ["-", a, b]]], eps],
    ],
    ">=": lambda a, b, eps: [
        [
            "-",
            1,
            [
                "/",
                ["+", ["-", b, a], ["abs", ["-", b, a]]],
                ["+", ["*", 2, ["abs", ["-", b, a]]], eps],
            ],
        ]
    ],
    "<": lambda a, b, eps: [
        "/",
        ["+", ["-", b, a], ["abs", ["-", b, a]]],
        ["+", ["*", 2, ["abs", ["-", b, a]]], eps],
    ],
    "<=": lambda a, b, eps: [
        "-",
        1,
        [
            "/",
            ["+", ["-", a, b], ["abs", ["-", a, b]]],
            ["+", ["*", 2, ["abs", ["-", a, b]]], eps],
        ],
    ],
    "!": lambda a: ["-", 1, a],
    "sign": lambda a, eps: ["/", a, ["+", ["abs", a], eps]],
    "max": lambda a, b: ["/", ["+", ["+", a, b], ["abs", ["-", a, b]]], 2],
    "max0": lambda a: ["/", ["+", a, ["abs", a]], 2],
    "min": lambda a, b: ["/", ["-", ["+", a, b], ["abs", ["-", a, b]]], 2],
    "min0": lambda a: ["/", ["-", a, ["abs", a]], 2],
    "if": lambda cond, then_val, else_val: [
        "+",
        ["*", then_val, cond],
        ["*", else_val, ["-", 1, cond]],
    ],
    "mod": lambda a, b: [
        "/",
        ["*", b, ["arctan", ["tan", ["/", ["*", "pi", a], b]]]],
        "pi",
    ],
    "frac": lambda a: ["/", ["arctan", ["tan", ["*", "pi", a]]], "pi"],
    "floor": lambda a: [
        "-",
        ["-", a, 0.5],
        ["/", ["arctan", ["tan", ["+", ["*", "pi", a], ["*", 0.5, "pi"]]]], "pi"],
    ],
    "ceil": lambda a: [
        "+",
        ["+", a, 0.5],
        ["/", ["arctan", ["tan", ["+", ["*", "pi", a], ["*", 0.5, "pi"]]]], "pi"],
    ],
    "round": lambda a: ["-", a, ["/", ["arctan", ["tan", ["*", "pi", a]]], "pi"]],
}
INFIX_OPERATIONS = {"==", "!=", ">", "<", ">=", "<=", "+", "-", "*", "/", "**"}
SYMPY_WORKERS = 4
precedence = {
    "==": 0,
    "!=": 0,
    ">": 0,
    "<": 0,
    ">=": 0,
    "<=": 0,
    "+": 1,
    "-": 1,
    "*": 2,
    "/": 2,
    "**": 3,
}
right_assoc = {"**", "+", "-", "*", "/"}
COMMUTATIVE_OPERATIONS = {"+", "*"}

math_funcs = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "arcsin": math.asin,
    "arccos": math.acos,
    "arctan": math.atan,
    "sqrt": math.sqrt,
    "abs": abs,
    "pi": math.pi,
    "euler": math.e,
}

EVAL_GLOBALS = {
    "__builtins__": builtins.__dict__,
    **math_funcs,
}

_MULTI_CHAR_OPS = [key for key in INFIX_OPERATIONS if len(key) > 1]
_MULTI_CHAR_OPS += [
    key
    for key in OPERATION_EXPANSIONS
    if len(key) > 1 and not key.isidentifier() and key not in _MULTI_CHAR_OPS
]
_MULTI_CHAR_OPS += [":=", "//", "**"]

_MULTI_OP_PLACEHOLDERS = {
    f"__MULTIOP_{i}__": op for i, op in enumerate(_MULTI_CHAR_OPS)
}
_OP_TO_PLACEHOLDER = {op: ph for ph, op in _MULTI_OP_PLACEHOLDERS.items()}
_MULTI_OP_PATTERN = (
    re.compile(
        "|".join(re.escape(op) for op in sorted(_MULTI_CHAR_OPS, key=len, reverse=True))
    )
    if _MULTI_CHAR_OPS
    else None
)

_SPLITTER_PATTERN = re.compile(r"([(),+\-*/=!\n])")
_TAB_PATTERN = re.compile(r"\t| {4}")
_SPACES_PATTERN = re.compile(r" +")


def _log_message(prefix: str, message: str, max_len: int = 127):
    original_len = len(message)
    truncated = message[:max_len]
    log_line = f"[{prefix} {original_len}] {truncated}{'...' if original_len > max_len else ''}"
    print(log_line)
    if LOGFILE:
        with open(LOGFILE, "a") as f:
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


def stable_serialize(obj) -> str:
    def _to_canonical_jsonable(x):
        if x is None or isinstance(x, (bool, int, float, str)):
            return x

        if isinstance(x, list):
            return [_to_canonical_jsonable(v) for v in x]

        if isinstance(x, tuple):
            return {"__tuple__": [_to_canonical_jsonable(v) for v in x]}

        if isinstance(x, dict):
            items = sorted(
                x.items(),
                key=lambda kv: stable_serialize(kv[0]),
            )
            return {
                "__dict__": [
                    [_to_canonical_jsonable(k), _to_canonical_jsonable(v)]
                    for k, v in items
                ]
            }

        if isinstance(x, set):
            return {
                "__set__": sorted(stable_serialize(v) for v in x),
            }

        if isinstance(x, complex):
            return {"__complex__": [x.real, x.imag]}

        return {"__repr__": repr(x)}

    canonical = _to_canonical_jsonable(obj)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=True)


def freeze_ast(obj, _memo=None):
    if _memo is None:
        _memo = {}

    cacheable = isinstance(obj, (list, tuple, dict, set))
    if cacheable:
        obj_id = id(obj)
        if obj_id in _memo:
            return _memo[obj_id]

    if obj is None:
        result = ("none",)
        return result

    if isinstance(obj, bool):
        result = ("bool", obj)
        return result

    if isinstance(obj, int):
        result = ("int", obj)
        return result

    if isinstance(obj, float):
        result = ("float", obj)
        return result

    if isinstance(obj, str):
        result = ("str", obj)
        return result

    if isinstance(obj, list):
        result = ("list", tuple(freeze_ast(v, _memo) for v in obj))
        _memo[obj_id] = result
        return result

    if isinstance(obj, tuple):
        result = ("tuple", tuple(freeze_ast(v, _memo) for v in obj))
        _memo[obj_id] = result
        return result

    if isinstance(obj, dict):
        frozen_items = [
            (freeze_ast(k, _memo), freeze_ast(v, _memo)) for k, v in obj.items()
        ]
        frozen_items.sort(key=lambda kv: kv[0])
        result = ("dict", tuple(frozen_items))
        _memo[obj_id] = result
        return result

    if isinstance(obj, set):
        result = ("set", tuple(sorted(freeze_ast(v, _memo) for v in obj)))
        _memo[obj_id] = result
        return result

    if isinstance(obj, complex):
        result = ("complex", obj.real, obj.imag)
        return result

    result = ("repr", repr(obj))
    return result


def num(val: Union[str, int, float]) -> Union[int, float]:
    try:
        if float(val).is_integer():
            return int(val)
        else:
            return float(val)
    except ValueError:
        raise ValueError(f"Cannot turn {val!r} into a number")


class EvalVars:
    def __init__(self, d):
        self.__dict__.update(d)


@dataclass
class Output:
    output: str = None
    ast: Union[list, float, int] = None
    config: "ProgramConfig" = None


@dataclass
class ProgramConfig:
    precision: int = None
    epsilon: float = None
    variables: list = None
    dupe: bool = True
    dupe_min_savings: int = 0
    simplify: bool = True
    sympy: bool = False
    output: str = None

    def __post_init__(self):
        if self.variables is None:
            self.variables = list("abcdefxyzm")


@dataclass
class FunctionDef:
    name: str
    params: list
    body: list


@dataclass
class AstOutput:
    output: str = None
    ast: Union[list, float, int] = None
    sympy: bool = False


def is_constant(expr):
    if isinstance(expr, list):
        return all(is_constant(subexpr) for subexpr in expr[1:])
    else:
        if expr in ("epsilon", "pi", "euler"):
            return True
        try:
            num(expr)
            return True
        except ValueError:
            return False


def fix_structure(ast):
    if isinstance(ast, list):
        if len(ast) == 1:
            return fix_structure(ast[0])
        else:
            return [fix_structure(elem) for elem in ast]
    else:
        return ast


def tokenize(line):
    line = _TAB_PATTERN.sub(" __TAB__ ", line)
    if _MULTI_OP_PATTERN:
        line = _MULTI_OP_PATTERN.sub(
            lambda m: f" {_OP_TO_PLACEHOLDER[m.group(0)]} ", line
        )
    line = _SPLITTER_PATTERN.sub(r" \1 ", line)
    for placeholder, op in _MULTI_OP_PLACEHOLDERS.items():
        line = line.replace(placeholder, op)
    line = _SPACES_PATTERN.sub(" ", line)
    tokens = line.strip().split(" ")
    print_debug(f"tokenize: {len(tokens)} tokens, head={_preview_sequence(tokens)}")
    return tokens


def generate_ast(tokens):
    tokens = [t for t in tokens if t is not None and t != "" and t != "\n"]

    print_debug(f"generate_ast: {len(tokens)} tokens, head={_preview_sequence(tokens)}")

    def is_callable_token(token) -> bool:
        return isinstance(token, str) and (
            token.isidentifier() or token in OPERATION_EXPANSIONS
        )

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
                and is_callable_token(token)
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

    def looks_like_prefix_ast(node) -> bool:
        if not isinstance(node, list) or not node:
            return False

        head = node[0]
        if not isinstance(head, str):
            return False

        if head in {"+", "-"} and len(node) == 2:
            return False

        if head in INFIX_OPERATIONS or head in OPERATION_EXPANSIONS or head == "!":
            if any(is_op(tok) for tok in node[1:]):
                return False
            return True

        return False

    def convert(node):
        if not isinstance(node, list):
            return node

        if not node:
            return node

        if looks_like_prefix_ast(node):
            return [node[0]] + [convert(arg) for arg in node[1:]]

        is_func_call = (
            len(node) >= 2
            and isinstance(node[0], str)
            and node[0].isidentifier()
            and not is_op(node[0])
        )

        has_top_level_infix = any(is_op(tok) for tok in node)
        has_top_level_infix_in_tail = any(is_op(tok) for tok in node[1:])

        if len(node) > 1 and (
            (isinstance(node[1], str) and node[1] in INFIX_OPERATIONS)
            or (not is_func_call and has_top_level_infix_in_tail)
            or (
                len(node) == 2
                and isinstance(node[0], str)
                and node[0] in {"+", "-", "!"}
            )
        ):
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


def expand_expression(ast):
    if isinstance(ast, list):
        if not ast:
            return ast
        if not isinstance(ast[0], list) and ast[0] in OPERATION_EXPANSIONS:
            expansion_func = OPERATION_EXPANSIONS[ast[0]]
            args = expansion_func.__code__.co_varnames

            expanded_args = [expand_expression(arg) for arg in ast[1:]]
            if "eps" in args:
                return expansion_func(*expanded_args, "epsilon")
            else:
                return expansion_func(*expanded_args)
        else:
            return [expand_expression(elem) for elem in ast]
    else:
        return ast


def expand_functions(ast, functions):
    if isinstance(ast, list):
        if not isinstance(ast[0], list):
            for func in functions:
                if ast[0] == func.name:
                    if len(ast[1:]) != len(func.params):
                        raise ValueError(
                            f"Function {func.name} expects {len(func.params)} arguments, got {len(ast[1:])}"
                        )
                    expanded_args = [
                        expand_functions(arg, functions) for arg in ast[1:]
                    ]
                    substituted = substitute_vars(
                        func.body, dict(zip(func.params, expanded_args))
                    )
                    return expand_functions(substituted, functions)

        return [expand_functions(elem, functions) for elem in ast]
    else:
        return ast


def ast_sort_key(x):
    is_const = is_constant(x)
    try:
        length = ast_node_count(x) if isinstance(x, list) else (0 if is_const else 9999)
    except Exception:
        length = 9999
    frozen = freeze_ast(x)
    return (0 if is_const else 1, length, frozen)


def normalize_ast(ast):
    if not isinstance(ast, list):
        return ast

    if not ast:
        return ast

    op = ast[0] if isinstance(ast[0], str) else None
    if op not in COMMUTATIVE_OPERATIONS:
        return ast

    ast = [normalize_ast(elem) for elem in ast]
    operands = ast[1:]
    operands.sort(key=ast_sort_key, reverse=True)
    return [op] + operands


def parse_pragma(line: str, config: ProgramConfig):
    pragma, value = line.split(" ", 1)
    if pragma == "precision":
        if not value.isdigit():
            raise ValueError(f"Precision must be an integer, got {value!r}")
        config.precision = int(value)
    elif pragma == "epsilon":
        try:
            float(value)
        except ValueError:
            raise ValueError(f"Epsilon must be a float, got {value!r}") from None
        config.epsilon = float(value)
    elif pragma == "variables":
        config.variables = [var.strip() for var in value.split(" ")]
    elif pragma == "dupe":
        parts = value.strip().split()
        if not parts:
            raise ValueError("Dupe requires true/false")
        config.dupe = parts[0].lower() == "true"
        if len(parts) >= 2:
            try:
                config.dupe_min_savings = int(parts[1])
            except ValueError:
                raise ValueError("Dupe min_savings must be an integer") from None
    elif pragma == "simplify":
        config.simplify = value.lower() == "true"
    elif pragma == "sympy":
        config.sympy = value.lower() == "true"
    else:
        raise ValueError(f"Unknown pragma: {pragma}")


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
                    detail = (
                        f": {generate_expression(node)}" if (VERBOSE or DEBUG) else ""
                    )
                    base_msg = f"Malformed expression{detail}"
                    raise SyntaxError(
                        (
                            f"Line {line_num}: {base_msg}"
                            if line_num is not None
                            else base_msg
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

    walk(ast)


def substitute_vars(tokens, subst_vars, _stack=None):
    if _stack is None:
        _stack = set()

    def _list_needs_flatten(values: list) -> bool:
        saw_head = False
        for value in values:
            if isinstance(value, str):
                if value in {"(", ")", ","}:
                    return True
                if saw_head and value in INFIX_OPERATIONS:
                    return True
            saw_head = True
        return False

    if isinstance(tokens, list):
        out = []
        for token in tokens:
            replaced = substitute_vars(token, subst_vars, _stack)

            if isinstance(replaced, list):
                if len(replaced) == 1 and not isinstance(replaced[0], list):
                    out.append(replaced[0])
                    continue

                if _list_needs_flatten(replaced):
                    out.extend(replaced)
                else:
                    out.append(replaced)
                continue

            out.append(replaced)

        return out

    if tokens in subst_vars:
        if tokens in _stack:
            raise ValueError(f"Recursive substitution detected for {tokens!r}")
        _stack.add(tokens)
        try:
            return substitute_vars(subst_vars[tokens], subst_vars, _stack)
        finally:
            _stack.remove(tokens)

    return tokens


def parse_expression(tokens, subst_vars, functions, line_num, progress_cb=None):
    _validate_balanced_parentheses(tokens, line_num=line_num)

    if line_num is not None:
        print_verbose(f"line {line_num}: parsing expression")
    print_debug(
        f"parse_expression: raw tokens={_preview_sequence(tokens, max_items=32)}"
    )

    if progress_cb is not None:
        try:
            progress_cb(None, "substituting vars")
        except Exception:
            pass
    result = substitute_vars(tokens, subst_vars)

    print_debug(
        "parse_expression: after substitution tokens="
        + _preview_sequence(result, max_items=32)
    )
    print_verbose("After substitution: " + str(result))
    try:
        if progress_cb is not None:
            try:
                progress_cb(None, "generating ast")
            except Exception:
                pass
        ast = generate_ast(result)
    except SyntaxError as e:
        msg = str(e).strip()
        if line_num is not None and not msg.lower().startswith("line "):
            msg = f"Line {line_num}: {msg}"
        raise SyntaxError(msg) from None
    print_verbose("Generated AST: " + str(ast))

    if isinstance(ast, list) and len(ast) == 0:
        msg = "Empty expression"
        if line_num is not None:
            msg = f"Line {line_num}: {msg}"
        raise SyntaxError(msg)

    try:
        if progress_cb is not None:
            try:
                progress_cb(None, "infix to postfix")
            except Exception:
                pass
        ast = infix_to_postfix(ast)
    except IndexError:
        msg = "Malformed expression"
        if line_num is not None:
            msg = f"Line {line_num}: {msg}"
        raise SyntaxError(msg) from None
    except SyntaxError as e:
        msg = str(e).strip()
        if line_num is not None and not msg.lower().startswith("line "):
            msg = f"Line {line_num}: {msg}"
        raise SyntaxError(msg) from None

    print_debug(f"parse_expression: postfix ast type={type(ast).__name__}")

    print_verbose("Postfix AST: " + str(ast))
    if progress_cb is not None:
        try:
            progress_cb(None, "expanding expression")
        except Exception:
            pass
    expr = expand_expression(ast)
    print_debug(f"parse_expression: expanded ast type={type(expr).__name__}")
    print_verbose("Expanded AST: " + str(expr))
    try:
        if progress_cb is not None:
            try:
                progress_cb(None, "expanding functions")
            except Exception:
                pass
        expr = expand_functions(expr, functions)
    except Exception as e:
        msg = str(e).strip()
        if line_num is not None and not msg.lower().startswith("line "):
            msg = f"Line {line_num}: {msg}"
        raise type(e)(msg) from None
    print_verbose("After function expansion: " + str(expr))

    if progress_cb is not None:
        try:
            progress_cb(None, "normalizing")
        except Exception:
            pass
    expr = normalize_ast(expr)
    print_debug(f"parse_expression: normalized ast type={type(expr).__name__}")
    print_verbose("Normalized AST: " + str(expr))
    return expr


def parse_function(
    func_tokens,
    lines,
    subst_vars,
    functions,
    name,
    params=None,
    declared_variables=None,
    config: ProgramConfig | None = None,
    *,
    line_nums: list[int] | None = None,
    header_line_num: int | None = None,
    progress_cb=None,
):
    local_scope = subst_vars.copy()
    local_config = replace(config) if config is not None else ProgramConfig()
    for line_no, tokens in enumerate(func_tokens):
        if progress_cb is not None:
            try:
                progress_cb(
                    None, f"function body line {line_no + 1}/{len(func_tokens)}"
                )
            except Exception:
                pass
        statement_line_num = (
            line_nums[line_no]
            if (line_nums is not None and line_no < len(line_nums))
            else None
        )
        if tokens and tokens[0] == "//":
            try:
                parse_pragma(" ".join(tokens[1:]), local_config)
            except Exception as e:
                msg = str(e).strip()
                if statement_line_num is not None and not msg.lower().startswith(
                    "line "
                ):
                    msg = f"Line {statement_line_num}: {msg}"
                raise type(e)(msg) from None
            continue

        if len(tokens) > 1 and tokens[1] == "=":
            _raise_if_conflicts_with_declared_variables(
                tokens[0],
                declared_variables,
                kind="Substitution variable",
                line_num=statement_line_num,
            )
            local_expr = parse_expression(
                tokens[2:],
                local_scope,
                functions,
                statement_line_num,
                progress_cb=progress_cb,
            )
            local_scope[tokens[0]] = _apply_line_transforms(
                local_expr,
                config=local_config,
                functions=functions,
                line_num=statement_line_num,
            )
        elif len(tokens) > 1 and tokens[1] == ":=":
            var_name = tokens[0]
            _raise_if_conflicts_with_declared_variables(
                var_name,
                declared_variables,
                kind="Substitution variable",
                line_num=statement_line_num,
            )
            expr = lines[line_no].split(":=", 1)[1].strip()
            try:
                result = eval(expr, EVAL_GLOBALS, {"vars": EvalVars(local_scope)})
            except Exception as e:
                msg = f"Python eval failed: {e}"
                if statement_line_num is not None:
                    msg = f"Line {statement_line_num}: {msg}"
                raise ValueError(msg) from None
            local_scope[var_name] = [str(result)]

        elif tokens and tokens[0] == "return":
            expr_tokens = tokens[1:]
            expr = parse_expression(
                expr_tokens,
                local_scope,
                functions,
                statement_line_num,
                progress_cb=progress_cb,
            )
            expr = fix_structure(expr)
            expr = _apply_line_transforms(
                expr,
                config=local_config,
                functions=functions,
                line_num=statement_line_num,
            )
            _validate_no_undefined_identifiers(
                expr,
                declared_variables=(declared_variables or []) + (params or []),
                line_num=statement_line_num,
            )
            return expr

        else:
            msg = f"Unknown statement in function \"{name}\": {' '.join(tokens)}"
            if statement_line_num is not None:
                msg = f"Line {statement_line_num}: {msg}"
            raise SyntaxError(msg)

    # no return statement
    msg = f'Function "{name}" missing return statement'
    if header_line_num is not None:
        msg = f"Line {header_line_num}: {msg}"
    raise SyntaxError(msg)


def parse_repeat(
    body_tokens,
    lines,
    times,
    repeat_var,
    subst_vars,
    variables,
    functions,
    config,
    *,
    outer_bindings=None,
    output_state=None,
    line_nums: list[int] | None = None,
    header_line_num: int | None = None,
    progress_cb=None,
):
    if outer_bindings is None:
        outer_bindings = {}

    if output_state is None:
        output_state = {"output": None}

    variables = variables or []

    if repeat_var is not None:
        _raise_if_conflicts_with_declared_variables(
            repeat_var,
            variables,
            kind="Iteration variable",
            line_num=header_line_num,
        )
        if repeat_var in subst_vars.keys() or repeat_var in outer_bindings:
            msg = f"Iteration variable {repeat_var!r} conflicts with an existing substitution or loop binding"
            if header_line_num is not None:
                msg = f"Line {header_line_num}: {msg}"
            raise ValueError(msg)

    def dedent_once(token_list):
        if token_list and token_list[0] == "__TAB__":
            return token_list[1:]
        return token_list

    def strip_all_tabs(token_list):
        while token_list and token_list[0] == "__TAB__":
            token_list = token_list[1:]
        return token_list

    def _coerce_repeat_times_to_int(times_value, *, line_num: int | None):
        if isinstance(times_value, bool):
            msg = "Repetition count must be an integer"
            if line_num is not None:
                msg = f"Line {line_num}: {msg}"
            raise ValueError(msg)

        if isinstance(times_value, int):
            return times_value

        if isinstance(times_value, float):
            if times_value.is_integer():
                return int(times_value)
            msg = "Repetition count must be an integer"
            if line_num is not None:
                msg = f"Line {line_num}: {msg}"
            raise ValueError(msg)

        if isinstance(times_value, str):
            try:
                f = float(times_value)
            except Exception:
                msg = "Repetition count must be a constant integer expression"
                if line_num is not None:
                    msg = f"Line {line_num}: {msg}"
                raise ValueError(msg) from None
            if f.is_integer():
                return int(f)
            msg = "Repetition count must be an integer"
            if line_num is not None:
                msg = f"Line {line_num}: {msg}"
            raise ValueError(msg) from None

        msg = "Repetition count must be a constant integer expression"
        if line_num is not None:
            msg = f"Line {line_num}: {msg}"
        raise ValueError(msg)

    results = []
    try:
        if config.precision is None:
            raise ValueError("Precision must be set before repeat statement")
        if config.epsilon is None:
            raise ValueError("Epsilon must be set before repeat statement")

        times_tokens = times if isinstance(times, list) else [times]
        times_ast = parse_expression(
            times_tokens,
            subst_vars | outer_bindings,
            functions,
            header_line_num,
            progress_cb=progress_cb,
        )
        simplified = simplify_literals_ast(times_ast, config)
        if isinstance(simplified, list):
            raise ValueError("Repetition count must be a constant integer expression")

        total_iterations = _coerce_repeat_times_to_int(
            simplified, line_num=header_line_num
        )
    except Exception as e:
        msg = str(e).strip()
        if header_line_num is not None and not msg.lower().startswith("line "):
            msg = f"Line {header_line_num}: {msg}"
        raise ValueError(msg) from None

    for itr in range(total_iterations):
        if progress_cb is not None:
            try:
                progress_cb(None, f"iteration {itr + 1}/{total_iterations}")
            except Exception:
                pass
        itr_bindings = (
            outer_bindings
            if repeat_var is None
            else outer_bindings | {repeat_var: str(itr)}
        )

        line_no = 0
        while line_no < len(body_tokens):
            tokens = strip_all_tabs(body_tokens[line_no])
            statement_line_num = (
                line_nums[line_no]
                if (line_nums is not None and line_no < len(line_nums))
                else None
            )
            if not tokens:
                line_no += 1
                continue

            if len(tokens) > 1 and tokens[1] == "=":
                _raise_if_conflicts_with_declared_variables(
                    tokens[0],
                    variables,
                    kind="Substitution variable",
                    line_num=statement_line_num,
                )
                sub_expr = parse_expression(
                    tokens[2:],
                    subst_vars | itr_bindings,
                    functions,
                    statement_line_num,
                    progress_cb=progress_cb,
                )
                subst_vars[tokens[0]] = _apply_line_transforms(
                    sub_expr,
                    config=config,
                    functions=functions,
                    line_num=statement_line_num,
                )
                line_no += 1
                continue

            if tokens[0] == "//":
                try:
                    parse_pragma(" ".join(tokens[1:]), config)
                except Exception as e:
                    msg = str(e).strip()
                    if statement_line_num is not None and not msg.lower().startswith(
                        "line "
                    ):
                        msg = f"Line {statement_line_num}: {msg}"
                    raise type(e)(msg) from None
                line_no += 1
                continue

            if len(tokens) > 1 and tokens[1] == ":=":
                var_name = tokens[0]
                _raise_if_conflicts_with_declared_variables(
                    var_name,
                    variables,
                    kind="Substitution variable",
                    line_num=statement_line_num,
                )
                expr = lines[line_no].split(":=", 1)[1].strip()
                try:
                    result = eval(
                        expr,
                        EVAL_GLOBALS,
                        {"vars": EvalVars(subst_vars | itr_bindings)},
                    )
                except Exception as e:
                    msg = f"Python eval failed: {e}"
                    if statement_line_num is not None:
                        msg = f"Line {statement_line_num}: {msg}"
                    raise ValueError(msg) from None
                subst_vars[var_name] = [str(result)]
                line_no += 1
                continue

            if tokens[0] == "repeat":
                if len(tokens) < 2:
                    msg = "Repeat requires at least a repetition count"
                    if statement_line_num is not None:
                        msg = f"Line {statement_line_num}: {msg}"
                    raise SyntaxError(msg)
                if (
                    len(tokens) == 3
                    and isinstance(tokens[2], str)
                    and tokens[2].isidentifier()
                ):
                    nested_times = tokens[1]
                    nested_repeat_var = tokens[2]
                else:
                    nested_times = tokens[1:]
                    nested_repeat_var = None

                nested_body_tokens = []
                nested_body_lines = []
                nested_line_nums = []
                next_line_no = line_no + 1

                while next_line_no < len(body_tokens) and lines[
                    next_line_no
                ].startswith("    "):
                    nested_body_lines.append(lines[next_line_no][4:])
                    nested_body_tokens.append(dedent_once(body_tokens[next_line_no]))
                    if line_nums is not None and next_line_no < len(line_nums):
                        nested_line_nums.append(line_nums[next_line_no])
                    next_line_no += 1

                results.extend(
                    parse_repeat(
                        nested_body_tokens,
                        nested_body_lines,
                        nested_times,
                        nested_repeat_var,
                        subst_vars,
                        variables,
                        functions,
                        config,
                        outer_bindings=itr_bindings,
                        output_state=output_state,
                        line_nums=nested_line_nums,
                        header_line_num=statement_line_num,
                        progress_cb=progress_cb,
                    )
                )

                line_no = next_line_no
                continue

            if tokens[0] == "out":
                if len(tokens) < 2:
                    msg = "Out requires a variable name"
                    if statement_line_num is not None:
                        msg = f"Line {statement_line_num}: {msg}"
                    raise SyntaxError(msg)
                output_var = tokens[1]
                if output_var in variables:
                    output_state["output"] = output_var
                else:
                    msg = f"Output variable {output_var!r} not declared in variables"
                    if statement_line_num is not None:
                        msg = f"Line {statement_line_num}: {msg}"
                    raise ValueError(msg)
                line_no += 1
                continue

            if tokens[0] == "return":
                expr_tokens = tokens[1:]
                expr = parse_expression(
                    expr_tokens,
                    subst_vars | itr_bindings,
                    functions,
                    statement_line_num,
                    progress_cb=progress_cb,
                )
                expr = fix_structure(expr)

                _validate_no_undefined_identifiers(
                    expr,
                    declared_variables=variables,
                    line_num=statement_line_num,
                )
                results.extend(
                    _finalize_output_line_ast(
                        expr,
                        output_var=output_state["output"],
                        config=config,
                        functions=functions,
                        line_num=statement_line_num,
                    )
                )
                output_state["output"] = None
                line_no += 1
                continue

            raise SyntaxError(
                (
                    f"Line {statement_line_num}: Unknown statement in repeat: {' '.join(tokens)}"
                    if statement_line_num is not None
                    else f"Unknown statement in repeat: {' '.join(tokens)}"
                )
            )

    return results


def parse_pm3_to_ast(code, *, progress: bool = False):
    print_verbose(f"starting parse_pm3_to_ast: {len(code)} lines")
    print_debug(f"parse_pm3_to_ast: progress={progress}")
    config = ProgramConfig()
    subst_vars = {}

    output_state = {"output": None}

    collecting_function = False
    collecting_loop = False

    funcs = []
    compiled = []

    iterator = enumerate(code, start=1)
    bar = None
    if progress:
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:
            tqdm = None

        total = len(code) if hasattr(code, "__len__") else None
        if tqdm is not None:
            bar = tqdm(iterator, total=total, desc="pm3", unit="line")
            iterator = bar

    current_stage = "scanning"
    current_substage = None
    current_preview = None

    def _render_postfix() -> str:
        parts = []
        if current_substage:
            parts.append(str(current_substage))
        if current_preview:
            parts.append(str(current_preview))
        return " | ".join(parts)

    def progress_update(stage=None, substage=None):
        nonlocal current_stage, current_substage
        if bar is None:
            return

        changed = False
        if stage is not None and stage != current_stage:
            current_stage = stage
            try:
                bar.set_description_str(f"pm3 - {current_stage}")
            except Exception:
                pass
            changed = True

        if substage is not None and substage != current_substage:
            current_substage = substage
            changed = True

        if changed:
            try:
                bar.set_postfix_str(_render_postfix())
            except Exception:
                pass

    for line_num, line in iterator:
        if bar is not None:
            try:
                preview = line.split("#", 1)[0].strip()
                if len(preview) > 60:
                    preview = preview[:57] + "..."
                current_preview = f"L{line_num}: {preview}"
                bar.set_postfix_str(_render_postfix())
            except Exception:
                pass

        line = line.split("#", 1)[0].rstrip()
        if not line:
            continue

        tokens = tokenize(line)
        print_debug(
            f"scan line {line_num}: head tokens={_preview_sequence(tokens, max_items=24)}"
        )
        # print(line, tokens)

        if collecting_function:
            if not tokens[0] == "__TAB__":
                collecting_function = False
                progress_update(f"parsing function {function_name}", "parsing body")
                print_verbose(f"parsing function '{function_name}'")
                current_function.body = parse_function(
                    function_tokens,
                    function_body,
                    subst_vars,
                    funcs,
                    function_name,
                    current_function.params,
                    config.variables,
                    config=replace(config),
                    line_nums=function_line_nums,
                    header_line_num=function_header_line_num,
                    progress_cb=progress_update,
                )
                funcs.append(current_function)
                print_verbose(
                    f"registered function '{function_name}' (params={len(current_function.params)})"
                )
            else:
                function_body.append(line[4:])
                function_tokens.append(tokens[1:])
                function_line_nums.append(line_num)
                continue

        if collecting_loop:
            if not tokens[0] == "__TAB__":
                collecting_loop = False
                progress_update("unwrapping repeat", "parsing body")
                print_verbose("unwrapping repeat")
                repeat_result = parse_repeat(
                    repeat_tokens,
                    repeat_body,
                    repeat_times,
                    repeat_var,
                    subst_vars,
                    config.variables,
                    funcs,
                    config,
                    output_state=output_state,
                    line_nums=repeat_line_nums,
                    header_line_num=repeat_header_line_num,
                    progress_cb=progress_update,
                )
                compiled.extend(repeat_result)
                config.output = output_state["output"]
                print_verbose(f"repeat produced {len(repeat_result)} outputs")
            else:
                repeat_body.append(line[4:])
                repeat_tokens.append(tokens[1:])
                repeat_line_nums.append(line_num)

        if not collecting_function and not collecting_loop:
            if tokens[0] == "__TAB__":
                raise ValueError(
                    f"Line {line_num}: Unexpected indentation outside function or loop"
                )

            if tokens[0] == "def":
                if collecting_loop:
                    raise ValueError(
                        f"Line {line_num}: Function definitions inside loops are not supported"
                    )
                collecting_function = True
                function_body = []
                function_tokens = []
                function_line_nums = []
                function_name = tokens[1]
                progress_update(f"collecting function {function_name}", "reading body")
                print_verbose(f"collecting function '{function_name}'")
                params = tokens[2:]
                current_function = FunctionDef(
                    name=function_name, params=params, body=[]
                )
                function_header_line_num = line_num
            elif tokens[0] == "repeat":
                collecting_loop = True
                progress_update("collecting repeat", "reading body")
                print_verbose("collecting repeat")
                repeat_body = []
                repeat_tokens = []
                repeat_line_nums = []

                if (
                    len(tokens) == 3
                    and isinstance(tokens[2], str)
                    and tokens[2].isidentifier()
                ):
                    repeat_times = tokens[1]
                    repeat_var = tokens[2]
                else:
                    repeat_times = tokens[1:]
                    repeat_var = None
                repeat_header_line_num = line_num

            elif tokens[0] == "out":
                progress_update("parsing main body", "output selection")
                output = tokens[1]
                if output in config.variables:
                    config.output = output
                    output_state["output"] = output
                    print_verbose(f"selected output variable: {output}")
                else:
                    raise ValueError(
                        f"Line {line_num}: Output variable {output!r} not declared in variables"
                    )

            elif tokens[0] == "return":
                progress_update("parsing main body", "parsing return")
                print_verbose(f"compiling return expression (line {line_num})")
                if config.precision is None:
                    raise ValueError(
                        f"Line {line_num}: Precision must be set before return statement"
                    )
                if config.epsilon is None:
                    raise ValueError(
                        f"Line {line_num}: Epsilon must be set before return statement"
                    )
                expr_tokens = tokens[1:]
                expr = parse_expression(
                    expr_tokens,
                    subst_vars,
                    funcs,
                    line_num,
                    progress_cb=progress_update,
                )
                expr = fix_structure(expr)

                _validate_no_undefined_identifiers(
                    expr,
                    declared_variables=config.variables,
                    line_num=line_num,
                )
                compiled.extend(
                    _finalize_output_line_ast(
                        expr,
                        output_var=output_state["output"],
                        config=config,
                        functions=funcs,
                        line_num=line_num,
                    )
                )
                config.output = None
                output_state["output"] = None
                print_verbose(f"return compiled: total outputs now {len(compiled)}")

            elif tokens[0] == "//":
                if not collecting_function and not collecting_loop:
                    progress_update("parsing config", "pragma")
                    print_verbose(f"parsing pragma (line {line_num})")
                    try:
                        parse_pragma(line.strip()[2:].strip(), config)
                    except Exception as e:
                        msg = str(e).strip()
                        if not msg.lower().startswith("line "):
                            msg = f"Line {line_num}: {msg}"
                        raise type(e)(msg) from None
                else:
                    raise ValueError(
                        f"Line {line_num}: Pragmas inside functions or loops are not supported"
                    )

            else:
                # non-function, non-pragma line
                if len(tokens) > 1 and tokens[1] == "=":
                    progress_update("parsing main body", "substitution")
                    _raise_if_conflicts_with_declared_variables(
                        tokens[0],
                        config.variables,
                        kind="Substitution variable",
                        line_num=line_num,
                    )
                    sub_expr = parse_expression(
                        tokens[2:],
                        subst_vars,
                        funcs,
                        line_num,
                        progress_cb=progress_update,
                    )
                    sub_expr = _apply_line_transforms(
                        sub_expr,
                        config=config,
                        functions=funcs,
                        line_num=line_num,
                    )
                    subst_vars[tokens[0]] = sub_expr
                    print_debug(f"substitution set: {tokens[0]!r} = {sub_expr!r}")
                elif len(tokens) > 1 and tokens[1] == ":=":
                    # python eval!
                    progress_update("parsing main body", "python eval")
                    var_name = tokens[0]
                    _raise_if_conflicts_with_declared_variables(
                        var_name,
                        config.variables,
                        kind="Substitution variable",
                        line_num=line_num,
                    )
                    expr = line.split(":=")[1].strip()
                    try:
                        result = eval(
                            expr, EVAL_GLOBALS, {"vars": EvalVars(subst_vars)}
                        )
                    except Exception as e:
                        raise ValueError(
                            f"Line {line_num}: Python eval failed: {e}"
                        ) from None
                    subst_vars[var_name] = [str(result)]
                    print_debug(f"python eval set: {var_name!r} = {str(result)!r}")
                else:
                    raise SyntaxError(
                        f"Line {line_num}: Unknown statement: {' '.join(tokens)}"
                    )

    if collecting_function:
        progress_update(f"parsing function {function_name}", "parsing body")
        current_function.body = parse_function(
            function_tokens,
            function_body,
            subst_vars,
            funcs,
            function_name,
            current_function.params,
            config.variables,
            config=replace(config),
            line_nums=function_line_nums,
            header_line_num=function_header_line_num,
            progress_cb=progress_update,
        )
        funcs.append(current_function)

    if collecting_loop:
        progress_update("unwrapping repeat", "parsing body")
        repeat_result = parse_repeat(
            repeat_tokens,
            repeat_body,
            repeat_times,
            repeat_var,
            subst_vars,
            config.variables,
            funcs,
            config,
            output_state=output_state,
            line_nums=repeat_line_nums,
            header_line_num=repeat_header_line_num,
            progress_cb=progress_update,
        )
        compiled.extend(repeat_result)
        config.output = output_state["output"]

    print_debug(f"config: {config}")
    print_debug(f"subst_vars: {subst_vars}")
    print_debug(f"funcs: {funcs}")

    progress_update("finalizing", "fixing structure")
    print_verbose(
        f"finalizing: {len(compiled)} compiled outputs, {len(funcs)} functions"
    )
    output = []
    for i, item in enumerate(compiled):
        output.append(fix_structure(item))

    if bar is not None:
        try:
            bar.set_postfix_str("done")
            bar.close()
        except Exception:
            pass

    return compiled


def generate_expression(ast):
    def needs_parens(parent_op, child_ast, is_right=False):
        if not isinstance(child_ast, list):
            return False
        if not isinstance(child_ast[0], str):
            return True

        child_op = child_ast[0]
        if child_op not in precedence:
            return False

        parent_prec = precedence.get(parent_op, 999)
        child_prec = precedence.get(child_op, 999)

        if child_prec < parent_prec:
            return True

        if child_prec == parent_prec:
            if is_right and parent_op not in right_assoc:
                return True

        return False

    def needs_parens_for_scalar(parent_op, child, is_right=False):
        if parent_op != "**" or is_right:
            return False
        if isinstance(child, (int, float)):
            return child < 0
        if isinstance(child, str):
            try:
                return float(child) < 0
            except Exception:
                return False
        return False

    def gen(ast):
        if isinstance(ast, list):
            if not ast:
                return ""
            if isinstance(ast[0], str):
                func_name = ast[0]
                if func_name in INFIX_OPERATIONS:
                    if len(ast) == 2:
                        arg_str = gen(ast[1])
                        if needs_parens(func_name, ast[1], True):
                            arg_str = f"({arg_str})"
                        return f"{func_name}{arg_str}"

                    args = []
                    for i, arg in enumerate(ast[1:]):
                        is_right_arg = i == len(ast[1:]) - 1
                        arg_str = gen(arg)
                        if needs_parens(
                            func_name, arg, is_right_arg
                        ) or needs_parens_for_scalar(func_name, arg, is_right_arg):
                            arg_str = f"({arg_str})"
                        args.append(arg_str)
                    return f" {func_name} ".join(args)
                else:
                    args = [gen(arg) for arg in ast[1:]]
                    return f"{func_name}({', '.join(args)})"
            else:
                return "(" + " ".join(gen(elem) for elem in ast) + ")"
        else:
            return str(ast)

    return gen(ast)


def simplify_literals_ast(ast, config):
    def _contains_symbol(node, symbol: str) -> bool:
        if node == symbol:
            return True
        if isinstance(node, list):
            return any(_contains_symbol(child, symbol) for child in node)
        return False

    if isinstance(ast, list):
        simplified = [simplify_literals_ast(elem, config) for elem in ast]
        if is_constant(simplified):
            expr_str = generate_expression(simplified)
            protect_zero_rounding = (
                isinstance(simplified, list)
                and bool(simplified)
                and simplified[0] == "+"
                and _contains_symbol(simplified, "epsilon")
            )
            try:
                result = eval(expr_str, EVAL_GLOBALS, {"epsilon": config.epsilon})
            except ZeroDivisionError:
                return 0
            if isinstance(result, complex):
                raise ValueError(
                    f"Complex result is not supported in simplify mode: {expr_str} -> {result}"
                )
            if isinstance(result, float):
                if config.precision is not None:
                    rounded = round(result, config.precision)
                    if not (protect_zero_rounding and rounded == 0.0 and result != 0.0):
                        result = rounded
                if result.is_integer():
                    result = int(result)
            return result
        else:
            return simplified
    else:
        return ast


def simplify_algebratic_identities(ast):
    def _is_real_number_node(node) -> bool:
        return isinstance(node, (int, float)) and not isinstance(node, bool)

    if isinstance(ast, list):
        simplified_children = [ast[0]] + [
            simplify_algebratic_identities(child) for child in ast[1:]
        ]
        ast = simplified_children

        if ast[0] == "*":
            if len(ast) == 3:
                if ast[1] == 0 or ast[2] == 0:
                    return 0
                if ast[1] == 1:
                    return ast[2]
                if ast[2] == 1:
                    return ast[1]
                if ast[1] == -1:
                    return ["-", ast[2]]
                if ast[2] == -1:
                    return ["-", ast[1]]
                if ast[1] == ast[2]:
                    return ["**", ast[1], 2]
                if isinstance(ast[2], list) and ast[2][0] == "+":
                    return ["+", ["*", ast[1], ast[2][1]], ["*", ast[1], ast[2][2]]]
                if isinstance(ast[1], list) and ast[1][0] == "+":
                    return ["+", ["*", ast[1][1], ast[2]], ["*", ast[1][2], ast[2]]]
                if (
                    isinstance(ast[1], list)
                    and ast[1][0] == "**"
                    and isinstance(ast[2], list)
                    and ast[2][0] == "**"
                ):
                    if ast[1][1] == ast[2][1]:
                        return ["**", ast[1][1], ["+", ast[1][2], ast[2][2]]]
                if (
                    isinstance(ast[2], list)
                    and ast[2][0] == "**"
                    and ast[1] == ast[2][1]
                ):
                    return ["**", ast[1], ["+", ast[2][2], 1]]
                if (
                    isinstance(ast[1], list)
                    and ast[1][0] == "**"
                    and ast[2] == ast[1][1]
                ):
                    return ["**", ast[2], ["+", ast[1][2], 1]]

        if ast[0] == "+":
            if len(ast) == 3:
                if ast[1] == 0:
                    return ast[2]
                if ast[2] == 0:
                    return ast[1]
                if isinstance(ast[1], list) and ast[1][0] == "-" and len(ast[1]) == 2:
                    if ast[1][1] == ast[2]:
                        return 0
                if isinstance(ast[2], list) and ast[2][0] == "-" and len(ast[2]) == 2:
                    if ast[2][1] == ast[1]:
                        return 0
                if ast[1] == ast[2]:
                    return ["*", 2, ast[1]]

        if ast[0] == "-":
            if len(ast) == 2:
                if ast[1] == 0:
                    return 0
                if isinstance(ast[1], list) and ast[1][0] == "-" and len(ast[1]) == 2:
                    return ast[1][1]
                if isinstance(ast[1], list) and ast[1][0] == "*" and ast[1][1] == -1:
                    return ast[1][2]
                if isinstance(ast[1], list) and ast[1][0] == "*" and ast[1][2] == -1:
                    return ast[1][1]
            if len(ast) == 3:
                if ast[2] == 0:
                    return ast[1]
                if ast[1] == 0:
                    return ["-", ast[2]]
                if ast[1] == ast[2]:
                    return 0

        if ast[0] == "/":
            if len(ast) == 3:
                if ast[1] == 0:
                    return 0
                if ast[2] == 1:
                    return ast[1]
                if ast[1] == ast[2]:
                    return 1
                if ast[2] == 0:
                    return 0
                if isinstance(ast[1], list) and ast[1][0] == "*":
                    if ast[1][1] == ast[2]:
                        return ast[1][2]
                    if ast[1][2] == ast[2]:
                        return ast[1][1]
                if (
                    isinstance(ast[1], list)
                    and ast[1][0] == "**"
                    and isinstance(ast[2], list)
                    and ast[2][0] == "**"
                ):
                    if ast[1][1] == ast[2][1]:
                        return ["**", ast[1][1], ["-", ast[1][2], ast[2][2]]]
                if (
                    isinstance(ast[2], list)
                    and ast[2][0] == "**"
                    and ast[1] == ast[2][1]
                ):
                    return ["**", ast[1], ["-", 1, ast[2][2]]]

        if ast[0] == "**":
            if len(ast) == 3:
                if ast[2] == 0:
                    return 1
                if ast[2] == 1:
                    return ast[1]
                if ast[1] == 0 and ast[2] != 0:
                    return 0
                if ast[1] == 1:
                    return 1
                if isinstance(ast[1], list) and ast[1][0] == "**":
                    return ["**", ast[1][1], ["*", ast[1][2], ast[2]]]
                if ast[2] == -1:
                    return ["/", 1, ast[1]]
                if (
                    _is_real_number_node(ast[1])
                    and _is_real_number_node(ast[2])
                    and ast[1] < 0
                    and 0 < ast[2] < 1
                ):
                    raise ValueError("Root of negative number in simplification")

        if ast[0] == "sqrt":
            if _is_real_number_node(ast[1]) and ast[1] < 0:
                raise ValueError("Square root of negative number in simplification")
            if ast[1] == 0:
                return 0
            if ast[1] == 1:
                return 1

        return ast

    else:
        return ast


def simplify_ast(ast, config):
    ast = simplify_literals_ast(ast, config)
    prev = None
    while prev != ast:
        prev = ast
        # slight performance hit, i dont care :3
        ast = simplify_algebratic_identities(ast)
    return ast


def _can_simplify_now(config: ProgramConfig) -> bool:
    return (
        bool(config.simplify)
        and config.precision is not None
        and config.epsilon is not None
    )


def _apply_line_transforms(
    ast, *, config: ProgramConfig, functions, line_num: int | None
):
    expr = ast

    if _can_simplify_now(config):
        expr = simplify_ast(expr, config)

    if config.sympy:
        try:
            import sympy  # noqa: F401
        except ImportError:
            raise ImportError(
                "SymPy is required for sympy output. Please install it via 'pip install sympy'."
            ) from None

        expr_str = generate_expression(expr)
        _, sympy_expr = _sympy_simplify_worker((0, expr_str))
        expr = parse_expression(tokenize(sympy_expr), {}, functions, line_num)
        expr = fix_structure(expr)

        if _can_simplify_now(config):
            expr = simplify_ast(expr, config)

    return expr


def _finalize_output_line_ast(
    ast,
    *,
    output_var,
    config: ProgramConfig,
    functions,
    line_num: int | None,
):
    expr = _apply_line_transforms(
        ast,
        config=config,
        functions=functions,
        line_num=line_num,
    )

    if config.dupe:
        pairs = extract_subexpressions(
            expr,
            output_var,
            min_savings=config.dupe_min_savings,
        )
    else:
        pairs = [(expr, output_var)]

    out = []
    for expr_ast, out_var in pairs:
        final_ast = _apply_line_transforms(
            expr_ast,
            config=config,
            functions=functions,
            line_num=line_num,
        )
        out.append(
            Output(
                output=out_var,
                ast=final_ast,
                config=replace(
                    config,
                    output=out_var,
                    simplify=False,
                    dupe=False,
                    sympy=False,
                ),
            )
        )

    return out


def all_sublists(expr):
    if isinstance(expr, list):
        yield expr
        for item in expr:
            yield from all_sublists(item)


def replace_in_ast(ast, pattern, replacement: str):
    if ast == pattern:
        return replacement
    if isinstance(ast, list):
        return [replace_in_ast(x, pattern, replacement) for x in ast]
    return ast


def expr_length(ast) -> int:
    return ast_node_count(ast)


def ast_node_count(ast) -> int:
    if isinstance(ast, list):
        total = 1
        for child in ast:
            total += ast_node_count(child)
        return total
    return 1


def find_beneficial_duplicates(ast, min_length: int = 4, min_savings: int = -999):
    if not isinstance(ast, list):
        return []

    seen = {}
    freeze_memo = {}

    def _traverse(node):
        if not isinstance(node, list):
            return 1
        size = 1
        for child in node:
            size += _traverse(child)
        if size >= min_length:
            key = freeze_ast(node, freeze_memo)
            if key in seen:
                seen[key][0] += 1
            else:
                seen[key] = [1, node, size]
        return size

    _traverse(ast)

    beneficial_dupes = []
    for count, obj, size in seen.values():
        if count > 1:
            savings = count * size - (size + 3 + (count - 1) * 3)
            if savings > min_savings:
                beneficial_dupes.append((savings, count * size, obj))

    beneficial_dupes.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return beneficial_dupes


def extract_subexpressions(ast, output_variable, min_savings):
    current_ast = ast
    beneficial_dupes = find_beneficial_duplicates(current_ast, min_savings=min_savings)

    if not beneficial_dupes:
        return [(current_ast, output_variable)]
    largest_dupe = beneficial_dupes[0][2]
    replaced_ast = replace_in_ast(current_ast, largest_dupe, "ans")
    return [(largest_dupe, "ans"), (replaced_ast, output_variable)]


def _normalize_for_sympy(expr: str) -> str:
    return (
        expr.replace("arcsin", "asin")
        .replace("arccos", "acos")
        .replace("arctan", "atan")
        .replace("abs(", "Abs(")
    )


def _denormalize_from_sympy(expr: str) -> str:
    expr = (
        expr.replace("asin", "arcsin")
        .replace("acos", "arccos")
        .replace("atan", "arctan")
        .replace("Abs", "abs")
    )
    return expr


def _format_expression(expr: str) -> str:
    return " ".join(tokenize(expr)).replace("( ", "(").replace(" )", ")")


def _sympy_simplify_worker(task: tuple[int, str]) -> tuple[int, str]:
    index, expr_str = task
    try:
        import sympy

        sympy_in = _normalize_for_sympy(expr_str)
        sympy_expr = sympy.sympify(sympy_in)
        simplified = sympy.simplify(sympy_expr)
        out = str(simplified).replace(".0e", "e")
        out = _denormalize_from_sympy(out)
        out = _format_expression(out)
        return index, out
    except ImportError:
        # let me be silly for an error message nobody's ever gonna get
        raise ImportError(
            "hi! i dont know what happened but you somehow passed the earlier sympy check, but sympy doesnt import.\n"
            "perhaps your install is fauly? qwq 3: i literally have NO idea how you got here"
        )
    except Exception:
        return index, expr_str


def simplify_sympy_multicore(
    expressions: list[str], *, show_progress: bool = True
) -> list[str]:
    if not expressions:
        return []

    print_verbose(
        f"sympy: simplifying {len(expressions)} expression(s) (multicore={len(expressions) > 1})"
    )

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    tasks = list(enumerate(expressions))
    results: list[str] = list(expressions)

    if len(tasks) == 1:
        idx, simplified = _sympy_simplify_worker(tasks[0])
        results[idx] = simplified
        return results

    max_workers = max(1, min(len(tasks), SYMPY_WORKERS))
    print_debug(f"sympy: using {max_workers} worker(s)")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_sympy_simplify_worker, task) for task in tasks]
        iterator = as_completed(futures)

        if tqdm is not None and show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="sympy", unit="expr")

        for fut in iterator:
            idx, simplified = fut.result()
            results[idx] = simplified

    return results


def process_asts(asts, *, progress: bool = False):
    print_verbose(f"processing {len(asts)} AST output(s)")
    bar = None
    iterator = enumerate(asts)
    if progress:
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:
            tqdm = None
        if tqdm is not None:
            bar = tqdm(total=len(asts), desc="pm3 - processing", unit="ast")

    def progress_stage(stage: str | None = None):
        if bar is None:
            return
        if stage is None:
            return
        try:
            bar.set_postfix_str(stage)
        except Exception:
            pass

    try:
        outputs_ast = []
        for i, output in iterator:
            progress_stage("building outputs")

            if output.output is None:
                print_debug(f"process_asts[{i}]: output variable is None")
            else:
                print_debug(f"process_asts[{i}]: output={output.output!r}")

            if output.config.simplify:
                progress_stage("simplifying")
                output.ast = simplify_ast(output.ast, output.config)
                line = AstOutput(
                    output=output.output, ast=output.ast, sympy=output.config.sympy
                )

            else:
                line = AstOutput(
                    output=output.output, ast=output.ast, sympy=output.config.sympy
                )

            if output.config.dupe:
                progress_stage("deduping")
                print_debug(
                    f"process_asts[{i}]: dedupe enabled (min_savings={output.config.dupe_min_savings})"
                )
                for extraction, variable in extract_subexpressions(
                    line.ast,
                    line.output,
                    min_savings=output.config.dupe_min_savings,
                ):
                    outputs_ast.append(
                        AstOutput(
                            output=variable, ast=extraction, sympy=output.config.sympy
                        )
                    )
            else:
                outputs_ast.append(line)

            if bar is not None:
                try:
                    bar.update(1)
                except Exception:
                    pass

        exprs = [generate_expression(item.ast) for item in outputs_ast]
        print_verbose(f"generated {len(exprs)} expression string(s)")

        if sum(item.sympy for item in outputs_ast):
            progress_stage("sympy")
            print_verbose("sympy enabled for some/all outputs")
            try:
                import sympy
            except ImportError:
                raise ImportError(
                    "SymPy is required for sympy output. Please install it via 'pip install sympy'."
                ) from None

            sympy_indexes = [i for i, item in enumerate(outputs_ast) if item.sympy]
            sympy_exprs = [exprs[i] for i in sympy_indexes]
            simplified_sympy_exprs = simplify_sympy_multicore(
                sympy_exprs, show_progress=progress
            )
            for i, simplified in zip(sympy_indexes, simplified_sympy_exprs):
                exprs[i] = simplified

        outputs = [(item.output, exprs[i]) for i, item in enumerate(outputs_ast)]
        print_verbose("processing complete")
        return outputs
    finally:
        if bar is not None:
            try:
                bar.set_postfix_str("done")
                bar.close()
            except Exception:
                pass


def code_to_lines(source):
    if isinstance(source, str):
        lines = source.splitlines(keepends=True)
    else:
        lines = list(source)

    expanded_lines = [
        part if i == 0 else " " * (len(line) - len(line.lstrip())) + part.lstrip()
        for line in lines
        for i, part in enumerate(line.split(";"))
    ]

    code = []
    curr_line = ""
    bracket_level = 0
    last_level_zero_line = 0
    for i, line in enumerate(expanded_lines):
        if bracket_level == 0:
            bracket_level += line.count("(") - line.count(")")
            if bracket_level < 0:
                raise SyntaxError(f"Line {i+1}: Unmatched closing parenthesis")
            if bracket_level > 0:
                curr_line += line.rstrip()
                last_level_zero_line = i + 1
            else:
                code.append(line.rstrip())
        else:
            curr_line += line.strip()
            bracket_level += line.count("(") - line.count(")")
            if bracket_level < 0:
                raise SyntaxError(f"Line {i+1}: Unmatched closing parenthesis")
            if bracket_level == 0:
                code.append(curr_line)
                curr_line = ""

    if bracket_level > 0:
        raise SyntaxError(
            f"EOF: Unmatched opening parenthesis starting at line {last_level_zero_line}"
        )

    return code


def file_to_lines(filename):
    with open(filename) as f:
        return code_to_lines(f.readlines())


def check_python_eval(code):
    check = [(i + 1, line) for i, line in enumerate(code) if ":=" in line]
    return check


if __name__ == "__main__":
    with open("mockup.pm3") as f:
        code = file_to_lines("mockup.pm3")
        asts = parse_pm3_to_ast(code, progress=True)
        outputs = process_asts(asts, progress=True)

        with open("math.txt", "w") as f:
            for output, expr in outputs:
                f.write(f"to {output}:\n")
                f.write(expr + "\n")
                print(f"to {output}:")
                print(expr)
