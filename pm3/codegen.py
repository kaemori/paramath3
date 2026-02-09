from __future__ import annotations

from .constants import INFIX_OPERATIONS, precedence, right_assoc


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

    def gen(node):
        if isinstance(node, list):
            if not node:
                return ""
            if isinstance(node[0], str):
                func_name = node[0]
                if func_name in INFIX_OPERATIONS:
                    args = []
                    for i, arg in enumerate(node[1:]):
                        is_right_arg = i == len(node[1:]) - 1
                        arg_str = gen(arg)
                        if needs_parens(func_name, arg, is_right_arg):
                            arg_str = f"({arg_str})"
                        args.append(arg_str)
                    return f" {func_name} ".join(args)
                else:
                    args = [gen(arg) for arg in node[1:]]
                    return f"{func_name}({', '.join(args)})"
            else:
                return "(" + " ".join(gen(elem) for elem in node) + ")"
        else:
            return str(node)

    return gen(ast)
