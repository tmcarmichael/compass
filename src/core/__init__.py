"""Cross-cutting primitives: types, constants, exceptions, features, timing."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("compass")
except PackageNotFoundError:
    __version__ = "dev"
