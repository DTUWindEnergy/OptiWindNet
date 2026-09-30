# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import re
from collections.abc import Mapping
from typing import Any, ClassVar, TypeVar

__all__ = ('BiMap',)

KT = TypeVar('KT')
VT = TypeVar('VT')
_MISSING: Any = object()


def make_handle(s):
    return re.sub(r'\W|^(?=\d)', '_', s)


class NodeTagger:
    # 50 digits, 'I' and 'l' were dropped
    alphabet = 'abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNOPQRSTUVWXYZ'
    value: ClassVar[dict[str, int]] = {c: i for i, c in enumerate(alphabet)}

    def __getattr__(self, b50):
        dec = 0
        digit_value = 1
        if b50[0] < 'α':
            for digit in b50[::-1]:
                dec += self.value[digit] * digit_value
                digit_value *= 50
            return dec
        else:
            # for greek letters, only single digit is implemented
            return ord('α') - ord(b50[0]) - 1

    def __getitem__(self, dec):
        if dec is None:
            return '∅'
        elif isinstance(dec, str):
            return dec
        b50 = []
        if dec >= 0:
            while True:
                dec, digit = divmod(dec, 50)
                b50.append(self.alphabet[digit])
                if dec == 0:
                    break
            return ''.join(b50[::-1])
        else:
            return chr(ord('α') + (abs(dec) - 1) % 25)


class BiMap(dict[KT, VT]):
    """One-to-one mapping that keeps its inverse in the plain dict ``inv``.

    Lookups on both directions are those of ``dict``. Writes go through the
    forward mapping, which keeps ``inv`` in sync; ``inv`` must be treated as
    read-only, since writing to it desynchronizes the two directions.

    Args:
      fwd: initial mapping; its values must be unique.

    Raises:
      ValueError: if a value would be mapped from two different keys.
    """

    __slots__ = ('inv',)
    inv: dict[VT, KT]

    def __init__(self, fwd: Mapping[KT, VT] | None = None) -> None:
        super().__init__(fwd or {})
        self.inv = {v: k for k, v in self.items()}
        if len(self.inv) != len(self):
            raise ValueError('BiMap values must be unique')

    def __setitem__(self, key: KT, val: VT) -> None:
        inv = self.inv
        other = inv.get(val, _MISSING)
        if other is not _MISSING and other != key:
            raise ValueError(f'value {val!r} is already mapped from {other!r}')
        old = self.get(key, _MISSING)
        if old is not _MISSING:
            del inv[old]
        super().__setitem__(key, val)
        inv[val] = key

    def __delitem__(self, key: KT) -> None:
        del self.inv[super().pop(key)]

    def pop(self, key: KT, default: Any = _MISSING) -> Any:
        val = super().pop(key, _MISSING)
        if val is _MISSING:
            if default is _MISSING:
                raise KeyError(key)
            return default
        del self.inv[val]
        return val

    def popitem(self) -> tuple[KT, VT]:
        key, val = super().popitem()
        del self.inv[val]
        return key, val

    def setdefault(self, key: KT, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    # pyrefly: ignore[bad-override]
    def update(self, other: Mapping[KT, VT], /) -> None:
        for key, val in other.items():
            self[key] = val

    def __ior__(self, other: Any) -> 'BiMap[KT, VT]':
        self.update(other)
        return self

    def clear(self) -> None:
        super().clear()
        self.inv.clear()

    def copy(self) -> 'BiMap[KT, VT]':
        new = BiMap.__new__(type(self))
        dict.update(new, self)
        new.inv = self.inv.copy()
        return new

    def __reduce__(self) -> tuple[Any, ...]:
        # the default dict-subclass protocol replays items through __setitem__
        # before ``inv`` exists
        return type(self), (dict(self),)
