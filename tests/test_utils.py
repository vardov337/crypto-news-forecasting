import json

import numpy as np

from cryptonews.utils import save_json, set_seed, sha256_file


def test_set_seed_is_reproducible():
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_sha256_file(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_save_json_keeps_cyrillic(tmp_path):
    path = save_json({"этап": "0"}, tmp_path / "sub" / "m.json")
    assert json.loads(path.read_text(encoding="utf-8")) == {"этап": "0"}
