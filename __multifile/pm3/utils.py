from __future__ import annotations

from typing import Union


def stable_serialize(obj) -> str:
    if obj is None or isinstance(obj, (bool, int, float, complex, str)):
        return repr(obj)

    if isinstance(obj, (list, tuple)):
        open_bracket, close_bracket = (
            ("[", "]") if isinstance(obj, list) else ("(", ")")
        )
        return open_bracket + ",".join(stable_serialize(x) for x in obj) + close_bracket

    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: stable_serialize(kv[0]))
        return (
            "{"
            + ",".join(f"{stable_serialize(k)}:{stable_serialize(v)}" for k, v in items)
            + "}"
        )

    if isinstance(obj, set):
        items = sorted((stable_serialize(x) for x in obj))
        return "set{" + ",".join(items) + "}"

    return repr(obj)


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
