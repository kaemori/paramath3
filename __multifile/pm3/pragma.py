from __future__ import annotations

from .config import ProgramConfig


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
