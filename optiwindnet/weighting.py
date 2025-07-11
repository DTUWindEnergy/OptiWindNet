# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import numpy as np

__all__ = ()


class Weight:
    @classmethod
    def blockage_xtra(cls, data):
        arc = data['arc'][data['root']]
        penalty = np.pi / (np.pi - arc) + 4 * arc / np.pi
        return data['length'] * penalty

    @classmethod
    def blockage(cls, data):
        arc = data['arc'][data['root']]
        penalty = np.pi / (np.pi - arc)
        return data['length'] * penalty
