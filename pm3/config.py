from __future__ import annotations

from dataclasses import dataclass
from typing import Union


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
    dupe: bool = None
    dupe_min_savings: int = 0
    simplify: bool = None
    sympy: bool = None
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
