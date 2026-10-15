"""mouldcheck: draft, undercut, wall-thickness and parting-line checks for moulded parts.

    python -m mouldcheck part.stl --pull 0 0 1 --out results/
"""
from . import checks  # noqa: F401
from .mesh import Mesh, read_stl, write_stl  # noqa: F401
from .report import analyse, text_summary  # noqa: F401

__version__ = "0.1.0"
