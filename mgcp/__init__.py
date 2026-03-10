import pathlib

__version__ = "1.0.0"

try:
	from . import binary  # type: ignore
	from . import dna  # type: ignore
	from . import utils  # type: ignore
except Exception:
	# If submodules require compiled extensions or external binaries, we
	# don't want `import mgcp` to raise for users who only need parts.
	pass

from importlib.metadata import version as _dist_version, PackageNotFoundError

def get_version() -> str:
	"""Return the installed package version when available, else fallback."""
	try:
		return _dist_version("mgcp")
	except PackageNotFoundError:
		return __version__

__all__ = ["get_version", "__version__"]