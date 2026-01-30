from dataclasses import dataclass
from paramath import ProgramConfig, FunctionDef

functions = [
    FunctionDef(
        name="better_mod",
        params=["a", "b"],
        body=["fractional = frac(a / b)", "return fractional * b"],
        config=ProgramConfig(
            precision=3,
            epsilon=1e-99,
            variables=["a", "b", "c", "d", "e", "f", "x", "y", "z", "m"],
            dupe=False,
            simplify=True,
            sympy=False,
        ),
    )
]


file = """
fractional = frac(a / b)
return fractional * b""".strip()


def tokenize(file):
    splitters = ["(", ")", ",", "+", "-", "*", "/", "="]
    for splitter in splitters:
        file = file.replace(splitter, f" {splitter} ")
    while "  " in file:
        file = file.replace("  ", " ")
    return file.split(" ")


# substitutions
substs = {}


def parse_line(line: str):
    if line[1] == "=":
        # subst definition
        var_name, operator, value = line.split(" ", 2)
        if operator == "=":
            substs[var_name] = value

        elif operator == ":=":
            # python eval!
            math_funcs = {
                "frac": lambda x: x - int(x),
            }
            substs[var_name] = str(eval(value, {"__builtins__": None}, math_funcs))
        else:
            raise ValueError(f"Unknown operator in substitution: {operator}")


for line in file.split("\n"):
    tokens = tokenize(line)
    print(tokens)
    parse_line(line)
