# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

# author, version, license, and long description
__version__ = '0.0.1'
__author__ = 'Mauricio Souza de Alencar'

__doc__ = """
`interarray` implements extensions to the Esau-Williams heuristic for the
capacitaded minimum spanning tree (CMST) problem.

https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/
"""

__license__ = 'LGPL-2.1-or-later'

try:  # pragma: no cover
    # version.py created when installing optiwindnet
    from optiwindnet import version

    __version__ = version.__version__
    __release__ = version.__version__
except BaseException:  # pragma: no cover
    pass
