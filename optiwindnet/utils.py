# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import inspect
import re
from collections import namedtuple

__all__ = ()


def make_handle(s):
    return re.sub(r'\W|^(?=\d)', '_', s)


def namedtuplify(namedtuple_typename='', **kwargs):
    NamedTuplified = namedtuple(namedtuple_typename, tuple(str(kw) for kw in kwargs))
    return NamedTuplified(**kwargs)


class Alerter:
    def __init__(self, where, varname):
        self.where = where
        self.varname = varname
        self.f_creation = inspect.stack()[1].frame

    def __call__(self, text):
        i = self.f_creation.f_locals[self.varname]
        function = inspect.stack()[1].function
        if self.where(i, function):
            print(f'[{i}|{function}] ' + text)
