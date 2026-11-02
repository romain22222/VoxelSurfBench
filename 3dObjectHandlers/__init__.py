"""Handlers for 3D object formats. One file per format, each defining an `Object3D` subclass (see 3dObject.py).

Handlers are discovered automatically: every module of this package (except 3dObject) is imported and its
`Object3D` subclasses are indexed by their `extensions`.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

_base = importlib.import_module("3dObjectHandlers.3dObject")
Object3D = _base.Object3D

_HANDLERS: dict[str, Object3D] = {}
_EXTENSIONS: list[str] = []
_MESH_EXTENSIONS: list[str] = []


def _discover() -> None:
    for info in pkgutil.iter_modules(__path__):
        if info.name == "3dObject":
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, Object3D) and obj is not Object3D and obj.__module__ == module.__name__:
                if not obj.extensions:
                    raise TypeError(f"{obj.__name__} must define a non-empty `extensions` tuple")
                handler = obj()
                for ext in obj.extensions:
                    _HANDLERS[ext.lower()] = handler
                    _EXTENSIONS.append(ext.lower())
                    if not getattr(obj, "is_volume", False):
                        _MESH_EXTENSIONS.append(ext.lower())


_discover()


def supported_extensions() -> list[str]:
    return list(_EXTENSIONS)


def mesh_extensions() -> list[str]:
    """Extensions suitable for a triangle-mesh GT or method output (not volume inputs)."""
    return list(_MESH_EXTENSIONS)


def get_handler(path_or_ext: str | Path) -> Object3D:
    text = str(path_or_ext)
    ext = text.lower() if text.startswith(".") and "/" not in text and "\\" not in text else Path(text).suffix.lower()
    if ext not in _HANDLERS:
        raise ValueError(f"No handler for '{ext}' (supported: {', '.join(_EXTENSIONS)})")
    return _HANDLERS[ext]
