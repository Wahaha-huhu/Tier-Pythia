"""Chain-retrieval generator, emitting Pythia token ids. Pure numpy, no model needed.

A sequence carries M distinct content tokens forming a chain c0 -> c1 -> ... -> c(M-1).
The M-1 bindings (c_i, c_{i+1}) are emitted in random order as key, value, SEP, then a
query as marker, hop, start, target. Loss is on the final target token only.

Tasks
  compose      forward retrieval, main pool, mix of hop 1 and hop 2..K by p_multi
  hop1         forward retrieval, main pool, hop 1 only (the prerequisite)
  hop2_only    forward retrieval, main pool, hop 2 only
  fresh_hop1   forward retrieval, FRESH disjoint pool, QUERY_B, hop 1 (reuse control)
  reverse      reverse retrieval, main pool, QUERY_REV, answer the key of a queried value

Set shuffle=True to break the chain relation while preserving token identities and
positions, which gives the content-shuffled floor.
"""

import numpy as np


class Generator:
    def __init__(self, task_cfg):
        self.t = task_cfg

    # -- helpers --------------------------------------------------------------
    def _chain(self, rng, pool):
        # M distinct content ids sampled without replacement from the pool
        return rng.choice(pool, size=self.t.chain_length, replace=False)

    def _emit_bindings(self, rng, chain, shuffle_values=False):
        """Return the binding-table token block and the value-slot position per binding index.

        value_slot_pos[i] is the absolute position of the value token of binding i
        (the token c_{i+1} of the pair (c_i, c_{i+1})), accounting for the bos offset.
        """
        M = self.t.chain_length
        off = 1 if self.t.prepend_bos else 0
        keys = chain[:-1]                       # c_0 .. c_{M-2}
        values = chain[1:].copy()               # c_1 .. c_{M-1}
        if shuffle_values:
            values = values[rng.permutation(M - 1)]   # break the chain
        perm = rng.permutation(M - 1)           # slot -> binding index
        inv = np.empty(M - 1, dtype=np.int64)
        inv[perm] = np.arange(M - 1)            # binding index -> slot
        block = np.empty(3 * (M - 1), dtype=np.int64)
        for slot, bidx in enumerate(perm):
            block[3 * slot + 0] = keys[bidx]
            block[3 * slot + 1] = values[bidx]
            block[3 * slot + 2] = self.t.SEP
        value_slot_pos = off + 3 * inv + 1      # value of binding i sits at 3*slot(i)+1
        key_slot_pos = off + 3 * inv + 0
        return block, value_slot_pos, key_slot_pos

    def _assemble(self, block, query_tokens):
        off = 1 if self.t.prepend_bos else 0
        L = self.t.seq_len
        seq = np.empty(L, dtype=np.int64)
        if self.t.prepend_bos:
            seq[0] = self.t.eos_id
        seq[off:off + len(block)] = block
        seq[off + len(block):] = query_tokens
        return seq

    # -- batch ----------------------------------------------------------------
    def batch(self, task, n, rng, shuffle=False):
        """Return a dict of numpy arrays describing a batch of n sequences."""
        t = self.t
        M = t.chain_length
        L = t.seq_len
        ids = np.empty((n, L), dtype=np.int64)
        target = np.empty(n, dtype=np.int64)
        value_slot = np.full(n, -1, dtype=np.int64)   # for the key-slot score (hop 1 forward)
        hop = np.ones(n, dtype=np.int64)

        forward = task in ("compose", "hop1", "hop2_only", "fresh_hop1")
        pool = t.fresh_ids if task == "fresh_hop1" else t.content_ids
        marker = t.QUERY_B if task == "fresh_hop1" else (t.QUERY_REV if task == "reverse" else t.QUERY_A)

        for r in range(n):
            chain = self._chain(rng, pool)
            block, vpos, kpos = self._emit_bindings(rng, chain, shuffle_values=(shuffle and forward))

            if forward:
                if task == "compose":
                    h = 1 if rng.random() >= t.p_multi else int(rng.integers(2, t.k_max + 1))
                elif task == "hop2_only":
                    h = 2
                else:
                    h = 1
                s = int(rng.integers(0, M - h))           # start index, target exists
                start_tok = chain[s]
                tgt = chain[s + h]
                q = np.array([marker, t.HOP[h], start_tok, tgt], dtype=np.int64)
                if h == 1:
                    value_slot[r] = vpos[s]               # value of binding s is c_{s+1}
                hop[r] = h
            else:  # reverse: query a value c_j, answer its key c_{j-1}
                if shuffle:
                    # break the relation by shuffling which key precedes the queried value
                    block, vpos, kpos = self._emit_bindings(rng, chain, shuffle_values=True)
                j = int(rng.integers(1, M))               # value index 1..M-1
                start_tok = chain[j]
                tgt = chain[j - 1]
                q = np.array([marker, t.HOP[1], start_tok, tgt], dtype=np.int64)
                value_slot[r] = kpos[j - 1]               # key position of binding j-1
                hop[r] = 1

            ids[r] = self._assemble(block, q)
            target[r] = tgt

        return {
            "input_ids": ids,
            "target": target,
            "pred_pos": t.pred_pos,
            "target_pos": t.target_pos,
            "value_slot_pos": value_slot,
            "hop": hop,
        }


def repeated_token_diag(task_cfg, n, block_len, rng):
    """Repeated-token diagnostic for the induction score.

    Each row is a block of distinct random content ids repeated once, so the previous
    occurrence of any second-half token is unambiguous. Returns ids [n, 2*block_len].
    """
    t = task_cfg
    pool = np.array(t.content_ids + t.fresh_ids, dtype=np.int64)
    ids = np.empty((n, 2 * block_len), dtype=np.int64)
    for r in range(n):
        s = rng.choice(pool, size=block_len, replace=False)
        ids[r, :block_len] = s
        ids[r, block_len:] = s
    return ids
