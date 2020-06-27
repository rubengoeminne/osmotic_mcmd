"""
osmotic_mcmd
A package to perform mcmd simulations of guest-loaded MOFs in the osmotic ensemble, based on the yaff MD engine
"""

# Add imports here
from .mcmd import *
from .utilities import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
