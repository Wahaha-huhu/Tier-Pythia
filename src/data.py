"""Synthetic chain-retrieval task, rendered in Pythia's existing vocabulary.

A chain  x_0 -> x_1 -> ... -> x_L  is sampled from L+1 DISTINCT existing token IDs.
The context lists every edge as  [key, val, SEP]  in shuffled order (unordered bindings).
The query is  [QUERY, HOP_k, x_start]  and the supervised answer is a single token.

  Hop-1 (lookup primitive):   QUERY HOP_1 x_i  ->  x_{i+1}
  Hop-2 (composition):        QUERY HOP_2 x_i  ->  x_{i+2}     (must route through x_{i+1})

Only the final answer token carries loss (target-only mask), exactly as in the toy.
Because every interior token appears once as a value and once as a key, Hop-2 genuinely
requires recovering the intermediate and using it to select the second binding.

All examples have identical length (3L + 4 tokens), so batches need no padding.
"""
import numpy as np
import torch


class ChainTask:
    def __init__(self, cfg):
        self.cfg = cfg
        self.L = cfg.chain_len
        self.QUERY, self.HOP1, self.HOP2, self.SEP = cfg.marker_ids

        # Build a fixed content pool of existing token IDs (no new embedding rows).
        rng = np.random.default_rng(cfg.pool_seed)
        marker_set = set(cfg.marker_ids)
        candidates = np.arange(cfg.pool_lo, cfg.pool_hi, dtype=np.int64)
        rng.shuffle(candidates)
        pool = [int(c) for c in candidates if int(c) not in marker_set][: cfg.content_pool_size]
        assert len(pool) == cfg.content_pool_size, "content pool too small for the given range"
        self.pool = np.array(pool, dtype=np.int64)
        # sequence length (constant)
        self.seq_len = 3 * self.L + 4

    def _one_example(self, rng, hop):
        chain = rng.choice(self.pool, size=self.L + 1, replace=False)  # distinct tokens
        edges = [(int(chain[i]), int(chain[i + 1])) for i in range(self.L)]
        order = rng.permutation(self.L)
        ctx = []
        for j in order:
            k, v = edges[j]
            ctx += [k, v, self.SEP]
        if hop == 1:
            i = int(rng.integers(0, self.L))            # 0 .. L-1
            start, B, C = int(chain[i]), -1, int(chain[i + 1])
            hop_marker = self.HOP1
        else:
            i = int(rng.integers(0, self.L - 1))        # 0 .. L-2  (so i+2 <= L)
            start, B, C = int(chain[i]), int(chain[i + 1]), int(chain[i + 2])
            hop_marker = self.HOP2
        seq = ctx + [self.QUERY, hop_marker, start] + [C]
        cand = sorted({int(x) for x in chain})          # distinct in-context content tokens
        return seq, B, C, cand

    def batch(self, batch_size, hop, rng):
        seqs, Bs, Cs, cands = [], [], [], []
        for _ in range(batch_size):
            s, B, C, cand = self._one_example(rng, hop)
            seqs.append(s); Bs.append(B); Cs.append(C); cands.append(cand)
        input_ids = torch.tensor(seqs, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        labels[:, -1] = input_ids[:, -1]                # supervise only the final answer token
        return dict(
            input_ids=input_ids,
            labels=labels,
            B_ids=torch.tensor(Bs, dtype=torch.long),
            C_ids=torch.tensor(Cs, dtype=torch.long),
            cand_ids=torch.tensor(cands, dtype=torch.long),  # [B, L+1], constant width
        )
