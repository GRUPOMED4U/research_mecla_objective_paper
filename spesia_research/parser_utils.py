"""
This module contains utility functions for parsing command-line arguments.
"""

from typing import Any


def coerce_scalar(s: str) -> Any:
    # Convenience: bare true/false/none
    """
    Convenience function to convert a string to a scalar type (int, float, bool, None)
    If the string is "true" or "false", it will be converted to a bool.
    If the string is "none" or "null", it will be converted to None.
    If the string contains a ".", "e", or "E", it will be converted to a float.
    Otherwise, it will be converted to an int if possible, or left as a string if not.

    Args:
        s (str): The string to convert.

    Returns:
        Any: The converted scalar type.
    """
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None

    # Try int/float
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except Exception:
        return s  # fallback to raw string


def deep_merge(dict1, dict2):
    """
    Recursively merges dict2 into dict1.
    Values in dict2 will overwrite values in dict1 for non-dict types.
    """
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge if both values are dictionaries
            merged[key] = deep_merge(merged[key], value)
        else:
            # Overwrite or add the value from dict2
            merged[key] = value
    return merged


def set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """
    Set a value in a nested dictionary using a dotted key.

    For example, if d = {} and dotted_key = "a.b.c", then set_nested(d, dotted_key, 1)
    will result in d = {"a": {"b": {"c": 1}}.

    Args:
        d (dict): The dictionary to modify.
        dotted_key (str): The dotted key to set the value for.
        value (Any): The value to set.

    Returns:
        None
    """
    parts = dotted_key.split(".")
    new_dict = value
    for p in parts[::-1]:
        if p.isdigit():
            p = int(p)
        new_dict = {p: new_dict}
    d = deep_merge(d, new_dict)
    return d


def parse_kv_list(kvs: list[str]) -> dict:
    """
    Parse a list of key-value pairs into a dictionary.

    Args:
        kvs (list[str]): A list of key-value pairs in the format "key=value".

    Returns:
        dict: A dictionary containing the parsed key-value pairs.

    Raises:
        SystemExit: If a key-value pair is malformed (e.g. lacks "=").
    """
    out: dict = {}
    for item in kvs:
        if "=" not in item:
            raise SystemExit(f"Invalid --set '{item}'. Expected key=value.")
        k, v = item.split("=", 1)
        out = set_nested(out, k.strip(), coerce_scalar(v.strip()))
    return out
