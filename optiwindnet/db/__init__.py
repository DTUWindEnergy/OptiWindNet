# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import datetime
from .modelv2 import open_database
from .storagev2 import L_from_nodeset, G_from_routeset, G_by_method, Gs_from_attrs

def _naive_utc_now():
    '''Get UTC time now as a timezone-free datetime instance.

    This does the equivalent of the deprecated datetime.datetime.utcnow().
    '''
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
