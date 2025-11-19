__all__ = ["parse_args"]


def __getattr__(name: str):
    if name == "parse_args":
        from .arguments import parse_args as _parse_args

        globals()[name] = _parse_args
        return _parse_args
    raise AttributeError(f"module {__name__} has no attribute {name}")
