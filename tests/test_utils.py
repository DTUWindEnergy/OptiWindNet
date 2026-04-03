from optiwindnet.utils import make_handle, namedtuplify, NodeTagger


# --- make_handle ---


def test_make_handle_basic():
    assert make_handle('hello world') == 'hello_world'


def test_make_handle_leading_digit():
    assert make_handle('3sites') == '_3sites'


def test_make_handle_special_chars():
    assert make_handle('a-b.c/d') == 'a_b_c_d'


def test_make_handle_clean_string():
    assert make_handle('simple') == 'simple'


# --- namedtuplify ---


def test_namedtuplify_basic():
    nt = namedtuplify('Params', a=1, b='hello', c=[3, 4])
    assert nt.a == 1
    assert nt.b == 'hello'
    assert nt.c == [3, 4]


def test_namedtuplify_with_typename():
    nt = namedtuplify('MyTuple', x=10, y=20)
    assert type(nt).__name__ == 'MyTuple'
    assert nt.x == 10
    assert nt.y == 20


def test_namedtuplify_single_field():
    nt = namedtuplify('Single', val=42)
    assert nt.val == 42
    assert len(nt) == 1


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
