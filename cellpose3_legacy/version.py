"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

from importlib.metadata import PackageNotFoundError, version
import sys
from platform import python_version
import torch

try:
    version = version("cellpose3_legacy")
except PackageNotFoundError:
    version = "unknown"

version_str = f"""
cellpose3_legacy version: \t{version} 
platform:                 \t{sys.platform} 
python version:           \t{python_version()} 
torch version:            \t{torch.__version__}"""
