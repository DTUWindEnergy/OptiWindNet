# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import datetime


def _naive_utc_now():
    """Get UTC time now as a timezone-free datetime instance.

    This does the equivalent of the deprecated datetime.datetime.utcnow().
    """
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
