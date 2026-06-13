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
        self.n_distractors = getattr(cfg, "n_distractors", 0)
        self.QUERY, self.HOP1, self.HOP2, self.SEP = cfg.marker_ids

        # Build a fixed content pool of existing token IDs (no new embedding rows).
        rng = np.random.default_rng(cfg.pool_seed)
        marker_set = set(cfg.marker_ids)
        candidates = np.arange(cfg.pool_lo, cfg.pool_hi, dtype=np.int64)
        candidates = candidates[~np.isin(candidates, list(marker_set))]
        rng.shuffle(candidates)
        self.pool = candidates[: cfg.content_pool_size].copy()
        assert len(self.pool) == cfg.content_pool_size, "content pool too small for the given range"
        # HELD-OUT pool: an equally sized set of REAL tokens disjoint from the training pool,
        # used by the generalization battery to test token-identity invariance.
        rest = candidates[cfg.content_pool_size:]
        self.held_out_pool = rest[: cfg.content_pool_size].copy()
        # sequence length for the default config (constant when L and n_distractors are fixed)
        self.seq_len = 3 * (self.L + self.n_distractors) + 4

    def _one_example(self, rng, hop, pool=None, L=None, n_distractors=None):
        pool = self.pool if pool is None else pool
        L = self.L if L is None else L
        n_distractors = self.n_distractors if n_distractors is None else n_distractors

        chain = rng.choice(pool, size=L + 1, replace=False)  # distinct tokens
        edges = [(int(chain[i]), int(chain[i + 1])) for i in range(L)]

        # distractor edges: extra [key,val] bindings whose tokens are OFF the chain, so they
        # never participate in a correct lookup. The model must learn to ignore them.
        ctx_edges = list(edges)
        if n_distractors > 0:
            chain_set = set(int(x) for x in chain)
            avail = pool[~np.isin(pool, list(chain_set))]
            need = 2 * n_distractors
            extra = rng.choice(avail, size=min(need, len(avail)), replace=False)
            for j in range(0, len(extra) - 1, 2):
                ctx_edges.append((int(extra[j]), int(extra[j + 1])))

        order = rng.permutation(len(ctx_edges))
        ctx = []
        for j in order:
            k, v = ctx_edges[j]
            ctx += [k, v, self.SEP]
        if hop == 1:
            i = int(rng.integers(0, L))                 # 0 .. L-1
            start, B, C = int(chain[i]), -1, int(chain[i + 1])
            hop_marker = self.HOP1
        else:
            i = int(rng.integers(0, L - 1))             # 0 .. L-2  (so i+2 <= L)
            start, B, C = int(chain[i]), int(chain[i + 1]), int(chain[i + 2])
            hop_marker = self.HOP2
        seq = ctx + [self.QUERY, hop_marker, start] + [C]
        cand = sorted({int(x) for x in chain})          # answer is always a chain token
        return seq, B, C, cand

    def batch(self, batch_size, hop, rng, pool=None, L=None, n_distractors=None):
        seqs, Bs, Cs, cands = [], [], [], []
        for _ in range(batch_size):
            s, B, C, cand = self._one_example(rng, hop, pool=pool, L=L, n_distractors=n_distractors)
            seqs.append(s); Bs.append(B); Cs.append(C); cands.append(cand)
        input_ids = torch.tensor(seqs, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        labels[:, -1] = input_ids[:, -1]                # supervise only the final answer token
        return dict(
            input_ids=input_ids,
            labels=labels,
            B_ids=torch.tensor(Bs, dtype=torch.long),
            C_ids=torch.tensor(Cs, dtype=torch.long),
            cand_ids=torch.tensor(cands, dtype=torch.long),  # [B, L+1], constant width per call
        )
