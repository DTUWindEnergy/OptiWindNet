# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

# author, version, license, and long description
__author__ = 'Mauricio Souza de Alencar'

__doc__ = """
Tool for designing and optimizing the electrical cable network (collection system)
for offshore wind power plants.

https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/
"""

__license__ = 'MIT'

try:  # pragma: no cover
    # version.py created when installing optiwindnet
    from optiwindnet import version

    __version__ = version.__version__
    __release__ = version.__version__
# `version.py` only exists in an installed tree; running from a source
# checkout without it simply leaves the version attributes undefined.
except ImportError:  # pragma: no cover
    pass
