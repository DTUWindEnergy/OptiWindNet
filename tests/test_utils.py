# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import copy
import pickle

import pytest

from optiwindnet.utils import BiMap, NodeTagger, make_handle

# --- make_handle ---


def test_make_handle_basic():
    assert make_handle('hello world') == 'hello_world'


def test_make_handle_leading_digit():
    assert make_handle('3sites') == '_3sites'


def test_make_handle_special_chars():
    assert make_handle('a-b.c/d') == 'a_b_c_d'


def test_make_handle_clean_string():
    assert make_handle('simple') == 'simple'


# --- NodeTagger ---


class TestNodeTagger:
    def setup_method(self):
        self.N = NodeTagger()

    def test_single_digit_encode(self):
        # 0 -> 'a', 1 -> 'b', ...
        assert self.N[0] == 'a'
        assert self.N[1] == 'b'
        assert self.N[49] == 'Z'

    def test_multi_digit_encode(self):
        # 50 -> 'ba' (1*50 + 0)
        assert self.N[50] == 'ba'
        # 51 -> 'bb' (1*50 + 1)
        assert self.N[51] == 'bb'

    def test_single_digit_decode(self):
        assert self.N.a == 0
        assert self.N.b == 1
        assert self.N.Z == 49

    def test_multi_digit_decode(self):
        assert self.N.ba == 50
        assert self.N.bb == 51

    def test_roundtrip(self):
        for i in range(200):
            encoded = self.N[i]
            decoded = getattr(self.N, encoded)
            assert decoded == i, f'roundtrip failed for {i}: encoded={encoded}'

    def test_none_gives_empty_set(self):
        assert self.N[None] == '∅'

    def test_string_passthrough(self):
        assert self.N['hello'] == 'hello'

    def test_negative_gives_greek(self):
        # -1 -> 'α', -2 -> 'β', etc.
        result = self.N[-1]
        assert result == 'α'
        result2 = self.N[-2]
        assert result2 == 'β'

    def test_greek_decode(self):
        # 'α' -> -1, 'β' -> -2
        assert self.N.α == -1
        assert self.N.β == -2


# --- BiMap ---


def _assert_synced(m):
    assert m.inv == {v: k for k, v in m.items()}


def test_bimap_rejects_duplicate_values():
    with pytest.raises(ValueError):
        BiMap({1: 'a', 2: 'a'})
    m = BiMap({1: 'a', 2: 'b'})
    with pytest.raises(ValueError):
        m[3] = 'a'
    _assert_synced(m)


def test_bimap_mutations_keep_inverse_synced():
    m = BiMap({1: 'a', 2: 'b', 3: 'c'})
    m[1] = 'z'
    m[1] = 'z'
    del m[2]
    assert m.pop(3) == 'c'
    assert m.pop(3, None) is None
    m.update({4: 'd'})
    m |= {5: 'e'}
    assert m.setdefault(6, 'f') == 'f'
    _assert_synced(m)
    assert m == {1: 'z', 4: 'd', 5: 'e', 6: 'f'}
    m.popitem()
    _assert_synced(m)
    m.clear()
    assert not m and not m.inv


def test_bimap_copies_are_independent():
    m = BiMap({(0, 1): (2, 3)})
    for c in (m.copy(), copy.copy(m), copy.deepcopy(m), pickle.loads(pickle.dumps(m))):
        assert type(c) is BiMap and c == m and c.inv == m.inv
        del c[(0, 1)]
        assert m.inv == {(2, 3): (0, 1)}
