"""Offline tests for the generator and the task config. No torch, run with python -m pytest
or directly with python tests/test_generator.py.
"""

import numpy as np

from cp_pythia.config import TaskConfig
from cp_pythia.generator import Generator, repeated_token_diag


def test_layout_and_positions():
    t = TaskConfig(content_vocab=64, chain_length=8, k_max=2)
    assert t.seq_len == 3 * (8 - 1) + 4 + 1          # bos + bindings + query
    assert t.pred_pos == t.seq_len - 2
    assert t.target_pos == t.seq_len - 1
    g = Generator(t)
    b = g.batch("compose", 32, np.random.default_rng(0))
    assert b["input_ids"].shape == (32, t.seq_len)
    # the supervised target equals the token at target_pos
    assert np.all(b["input_ids"][:, t.target_pos] == b["target"])
    # bos present
    assert np.all(b["input_ids"][:, 0] == t.eos_id)


def test_distinct_chain_and_specials():
    t = TaskConfig(content_vocab=64, chain_length=8, k_max=2)
    g = Generator(t)
    b = g.batch("hop1", 64, np.random.default_rng(1))
    off = 1
    # the binding block has SEP at every third position
    sep_positions = off + 2 + 3 * np.arange(t.chain_length - 1)
    assert np.all(b["input_ids"][:, sep_positions] == t.SEP)
    # query marker is QUERY_A for the main pool
    qpos = off + 3 * (t.chain_length - 1)
    assert np.all(b["input_ids"][:, qpos] == t.QUERY_A)


def test_keyslot_position_holds_the_value():
    # for a hop-1 forward query, the recorded value_slot_pos must hold a content token
    # that is the correct answer, since following the queried key gives the target
    t = TaskConfig(content_vocab=64, chain_length=8, k_max=2)
    g = Generator(t)
    rng = np.random.default_rng(2)
    b = g.batch("hop1", 200, rng)
    vsp = b["value_slot_pos"]
    assert np.all(vsp >= 0)
    rows = np.arange(b["input_ids"].shape[0])
    token_at_value_slot = b["input_ids"][rows, vsp]
    # for hop 1 the value slot holds c_{s+1}, which is exactly the target
    assert np.all(token_at_value_slot == b["target"])


def test_shuffle_breaks_the_relation():
    t = TaskConfig(content_vocab=64, chain_length=8, k_max=2)
    g = Generator(t)
    rng = np.random.default_rng(3)
    b = g.batch("hop1", 500, rng, shuffle=True)
    vsp = b["value_slot_pos"]
    rows = np.arange(b["input_ids"].shape[0])
    token_at_value_slot = b["input_ids"][rows, vsp]
    # after shuffling the value column the value slot rarely equals the true target
    frac_match = np.mean(token_at_value_slot == b["target"])
    assert frac_match < 0.5, frac_match


def test_fresh_pool_disjoint_and_marker():
    t = TaskConfig(content_vocab=64, chain_length=8, k_max=2)
    g = Generator(t)
    b = g.batch("fresh_hop1", 64, np.random.default_rng(4))
    off = 1
    qpos = off + 3 * (t.chain_length - 1)
    assert np.all(b["input_ids"][:, qpos] == t.QUERY_B)
    # all content tokens come from the fresh pool
    fresh = set(t.fresh_ids)
    body = b["input_ids"][:, off:qpos]
    content = body[(body != t.SEP)]
    assert set(np.unique(content)).issubset(fresh)


def test_reverse_target_is_the_key():
    t = TaskConfig(content_vocab=64, chain_length=8, k_max=2)
    g = Generator(t)
    b = g.batch("reverse", 200, np.random.default_rng(5))
    off = 1
    qpos = off + 3 * (t.chain_length - 1)
    assert np.all(b["input_ids"][:, qpos] == t.QUERY_REV)
    assert np.all(b["input_ids"][:, t.target_pos] == b["target"])


def test_repeated_diag():
    t = TaskConfig(content_vocab=64, chain_length=8)
    ids = repeated_token_diag(t, 16, 12, np.random.default_rng(6))
    assert ids.shape == (16, 24)
    # the two halves are identical
    assert np.all(ids[:, :12] == ids[:, 12:])
    # within a row the block is distinct
    for r in range(ids.shape[0]):
        assert len(set(ids[r, :12])) == 12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all generator tests passed")
