import math
import builtins
from dataclasses import dataclass
from typing import Union


def num(val: Union[str, int, float]) -> Union[int, float]:
    try:
        if float(val).is_integer():
            return int(val)
        else:
            return float(val)
    except ValueError:
        raise ValueError(f"Cannot turn {val!r} into a number")


@dataclass
class ProgramConfig:
    precision: int = None
    epsilon: float = None
    variables: list = None
    dupe: bool = None
    simplify: bool = None
    sympy: bool = None


@dataclass
class FunctionDef:
    name: str
    params: list
    body: list
    config: ProgramConfig


def parse_pragma(line: str, config: ProgramConfig):
    pragma, value = line.split(" ", 1)
    if pragma == "precision":
        config.precision = int(value)
    elif pragma == "epsilon":
        config.epsilon = float(value)
    elif pragma == "variables":
        config.variables = [var.strip() for var in value.split(" ")]
    elif pragma == "dupe":
        config.dupe = value.lower() == "true"
    elif pragma == "simplify":
        config.simplify = value.lower() == "true"
    elif pragma == "sympy":
        config.sympy = value.lower() == "true"
    else:
        raise ValueError(f"Unknown pragma: {pragma}")


def parse_global(line: str, global_vars: dict):
    var_name, operator, value = line.split(" ", 3)
    if operator == ":=":
        # python eval!
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
            "abs": abs,
            "pi": math.pi,
            "e": math.e,
        }

        eval_namespace = {
            "__builtins__": builtins.__dict__,
            "globals": global_vars,
            **math_funcs,
        }
        global_vars[var_name] = eval(value, eval_namespace)

    elif operator == "=":
        global_vars[var_name] = num(value)

    else:
        raise ValueError(f"Unknown operator in global declaration: {operator}")


def parse_pm3(code):
    config = ProgramConfig()
    global_vars = {}

    collecting_function = False
    collecting_loop = False

    funcs = []

    for line in code:
        if collecting_function:
            if not line.startswith("    "):
                collecting_function = False
                funcs.append(current_function)
            else:
                current_function.body.append(line[4:])
                continue

        if line.startswith("def"):
            if collecting_loop:
                raise ValueError("Function definitions inside loops are not supported")
            collecting_function = True
            name = line[3:].strip().split(":")[0].strip()
            params = [
                param.strip() for param in line[3:].strip().split(":")[1].split(",")
            ]
            current_function = FunctionDef(
                name=name, params=params, body=[], config=config
            )

        if line.strip().startswith("//"):
            if not collecting_function and not collecting_loop:
                parse_pragma(line.strip()[2:].strip(), config)
            else:
                raise ValueError("Pragmas inside functions or loops are not supported")
        if line.strip().startswith("global"):
            if not collecting_function and not collecting_loop:
                parse_global(line.strip()[6:].strip(), global_vars)
            else:
                raise ValueError(
                    "Global declarations inside functions or loops are not supported"
                )

    print(config)
    print(global_vars)
    print(funcs)


with open("mockup.pm3") as f:
    code = [line.rstrip() for line in f.readlines() if line]
    parse_pm3(code)
