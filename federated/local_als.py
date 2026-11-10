"""Module 3 — vectorised FedAvg building blocks (SPEC.md Section 4).

Every solve is  x = (A^T A + reg_param * n * I)^-1 A^T y  (Spark ALS's count-scaled regularization).

Everything is an ARRAY BLOCK per partition (never one Python record per user, item or rating):
  blocks : (p, (users[U], clients[U], offsets[U+1], items[N], ys[N]))  user partition p = user % P
  routes : (q, (keys[R], pids[R]))      item partition q = item % P; key = item (global V) or
                                        item * C + client (client-local versions, local epochs > 1)
  V      : (q, (keys[n], vecs[n, rank]))
One round (local_epochs = 1):
  deliver     V rows -> the user partitions that need them      (download; <= P blocks per q)
  user_pass   batched user solves, (client, item) grouping       (no data movement)
  item_pass   upload shuffle -> batched per-client solves, average by item, new V blocks in place
All worker-side code is built inside closures (shipped by value).
"""
from pyspark.sql import DataFrame

ROW_CHUNK = 65536  # rows per batched matmul: bounds temporary memory (rows x rank doubles)


def make_identity():
    """Partition function for integer keys (partition = key mod P). One object per run, since
    Spark treats RDDs as co-partitioned only when they share the same function object."""
    def ident(k):
        return k
    return ident


def _check_aligned(rdd, num_partitions: int, what: str):
    if rdd.getNumPartitions() != num_partitions:
        raise RuntimeError(f"{what}: inputs are not co-partitioned ({rdd.getNumPartitions()} partitions, "
                           f"expected {num_partitions}); the local join would be wrong")
    return rdd


def _make_kernels(reg_param: float, rank: int):
    """Vectorised normal equations and batched solves, as closures."""
    reg, r = float(reg_param), int(rank)

    def group_normal_eq(X, y, starts):
        """For row groups starting at `starts` (sorted, non-empty groups): X^T X, X^T y and count
        per group. Groups are bucketed by size, so each bucket is one batched BLAS matmul
        (k groups x m rows x rank) instead of per-row outer products."""
        import numpy as np
        n = len(y)
        G = len(starts)
        if G == 0:
            return np.zeros((0, r, r)), np.zeros((0, r)), np.zeros(0, dtype=np.int64)
        ends = np.append(starts[1:], n)
        sizes = (ends - starts).astype(np.int64)
        XtX = np.empty((G, r, r)); Xty = np.empty((G, r))
        for m in np.unique(sizes).tolist():
            idx = np.flatnonzero(sizes == m)
            for c0 in range(0, len(idx), max(1, ROW_CHUNK // m)):     # bound memory per batch
                sel = idx[c0:c0 + max(1, ROW_CHUNK // m)]
                rows = starts[sel][:, None] + np.arange(m)[None, :]
                B = X[rows]                                            # (k, m, rank)
                Bt = B.transpose(0, 2, 1)
                XtX[sel] = np.matmul(Bt, B)
                Xty[sel] = np.matmul(Bt, y[rows][:, :, None])[:, :, 0]
        return XtX, Xty, sizes

    def solve_batch(XtX, Xty, cnt):
        import numpy as np
        A = XtX + (reg * cnt)[:, None, None] * np.eye(r)
        return np.linalg.solve(A, Xty[:, :, None])[:, :, 0]

    def group_starts(sorted_keys):
        import numpy as np
        if len(sorted_keys) == 0:
            return np.zeros(0, dtype=np.int64)
        return np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]).astype(np.int64)

    return group_normal_eq, solve_batch, group_starts


def _make_splitter():
    """split(dest, *arrays) -> [(d, sub_arrays)] for each distinct destination, via one sort."""
    def split(dest, *arrays):
        import numpy as np
        if len(dest) == 0:
            return []
        order = np.argsort(dest, kind="stable")
        d = dest[order]
        cuts = np.flatnonzero(np.r_[True, d[1:] != d[:-1]])
        bounds = np.r_[cuts, len(d)]
        out = []
        for k in range(len(cuts)):
            sl = order[bounds[k]:bounds[k + 1]]
            out.append((int(d[bounds[k]]), tuple(a[sl] for a in arrays)))
        return out
    return split


def build_user_blocks(ratings_df: DataFrame, num_partitions: int, ident):
    """One block per user partition p (= user_idx mod P), sorted by user. Built once."""
    P = int(num_partitions)

    def to_block(pid, it):
        import numpy as np
        rows = list(it)
        if not rows:
            return
        u = np.fromiter((x[0] for x in rows), dtype=np.int64, count=len(rows))
        c = np.fromiter((x[1][0] for x in rows), dtype=np.int64, count=len(rows))
        i = np.fromiter((x[1][1] for x in rows), dtype=np.int64, count=len(rows))
        y = np.fromiter((x[1][2] for x in rows), dtype=np.float64, count=len(rows))
        order = np.argsort(u, kind="stable")
        u, c, i, y = u[order], c[order], i[order], y[order]
        starts = np.flatnonzero(np.r_[True, u[1:] != u[:-1]])
        offsets = np.r_[starts, len(u)].astype(np.int64)
        yield pid, (u[starts], c[starts], offsets, i, y)

    return (ratings_df.select("user_idx", "client_id", "item_idx", "target").rdd
            .map(lambda r: (int(r[0]), (int(r[1]), int(r[2]), float(r[3]))))
            .partitionBy(P, ident)
            .mapPartitionsWithIndex(to_block, preservesPartitioning=True))


def build_routes(blocks, num_partitions: int, ident, num_clients: int, per_client: bool):
    """(q, (keys[], pids[])) sorted by key: which user partitions need each item row
    (per_client=True: each (client, item) row, key = item * C + client)."""
    P, C = int(num_partitions), int(num_clients)
    split = _make_splitter()

    def emit(it):
        import numpy as np
        for pid, (users, clients, offsets, items, ys) in it:
            if per_client:
                keys = np.unique(items * C + np.repeat(clients, np.diff(offsets)))
                q = (keys // C) % P
            else:
                keys = np.unique(items)
                q = keys % P
            for dq, (k,) in split(q, keys):
                yield dq, (k, np.full(len(k), pid, dtype=np.int64))

    def combine(it):
        import numpy as np
        parts = [v for _q, v in it]
        if not parts:
            return
        keys = np.concatenate([a for a, _ in parts]); pids = np.concatenate([b for _, b in parts])
        order = np.lexsort((pids, keys))
        q = int((keys[0] // C) % P) if per_client else int(keys[0] % P)
        yield q, (keys[order], pids[order])

    return blocks.mapPartitions(emit).partitionBy(P, ident).mapPartitions(combine, preservesPartitioning=True)


def init_item_factors(routes, rank: int, std: float, seed: int):
    """(q, (item_ids[], V)) seeded N(0, std) per item_idx, rounded through float32 (as the NumPy
    reference); one generator per item so the result does not depend on partitioning."""
    rank, std, seed = int(rank), float(std), int(seed)

    def gen(it):
        import numpy as np
        for q, (keys, pids) in it:
            ids = np.unique(keys)
            V = np.empty((len(ids), rank))
            for k, i in enumerate(ids.tolist()):
                V[k] = np.random.default_rng([seed, int(i)]).normal(0.0, std, rank).astype(np.float32)
            yield q, (ids, V)

    return routes.mapPartitions(gen, preservesPartitioning=True)


def deliver(v, routes, ident, num_partitions: int):
    """The download: each item partition sends each user partition one block with exactly the rows
    it needs. V and routes are co-partitioned, so the lookup is local."""
    split = _make_splitter()
    both = _check_aligned(v.mapValues(lambda x: (0, x)).union(routes.mapValues(lambda x: (1, x))),
                          num_partitions, "deliver")

    def send(it):
        import numpy as np
        vk = vv = rk = rp = None
        for _q, (tag, x) in it:
            if tag == 0:
                vk, vv = x
            else:
                rk, rp = x
        if rk is None or len(rk) == 0:
            return
        pos = np.searchsorted(vk, rk)
        if (pos >= len(vk)).any() or (vk[np.minimum(pos, len(vk) - 1)] != rk).any():
            raise RuntimeError("deliver: a routed key has no V row in its partition (misalignment)")
        rows = vv[pos]
        for pid, (k, r_) in split(rp, rk, rows):
            yield pid, (k, r_)

    return both.mapPartitions(send).partitionBy(int(num_partitions), ident)


def _make_receive():
    def receive(it):
        """Split a co-partitioned (user block + delivered blocks) partition."""
        import numpy as np
        block, ks, rs = None, [], []
        for _p, (tag, x) in it:
            if tag == 0:
                block = x
            else:
                ks.append(x[0]); rs.append(x[1])
        if block is None:
            return None, None, None
        keys = np.concatenate(ks); rows = np.vstack(rs)
        order = np.argsort(keys, kind="stable")
        return block, keys[order], rows[order]
    return receive


def _make_user_solver():
    def solve_users_block(block, keys, rows, local, C, group_normal_eq, solve_batch):
        import numpy as np
        users, clients, offsets, items, ys = block
        cnt = np.diff(offsets)
        clients_r = np.repeat(clients, cnt)
        need = items * C + clients_r if local else items
        X = rows[np.searchsorted(keys, need)]
        XtX, Xty, n = group_normal_eq(X, ys, offsets[:-1])
        U = solve_batch(XtX, Xty, n)
        return U, cnt, clients_r
    return solve_users_block


def user_pass(blocks, rows, reg_param: float, rank: int, num_partitions: int, num_clients: int,
              local: bool):
    """Batched user solves; then, per (client, item): summed statistics if it has >= rank ratings
    in this partition, else its rows. One upload block per destination item partition q."""
    group_normal_eq, solve_batch, group_starts = _make_kernels(reg_param, rank)
    split, receive = _make_splitter(), _make_receive()
    P, C, r = int(num_partitions), int(num_clients), int(rank)
    solve_users_block = _make_user_solver()
    both = _check_aligned(blocks.mapValues(lambda b: (0, b)).union(rows.mapValues(lambda x: (1, x))),
                          P, "user_pass")

    def run(it):
        import numpy as np
        block, keys, vrows = receive(it)
        if block is None:
            return
        U, cnt, clients_r = solve_users_block(block, keys, vrows, local, C, group_normal_eq, solve_batch)
        items, ys = block[3], block[4]
        Ur = np.repeat(U, cnt, axis=0)
        gk = items * C + clients_r                               # (client, item) group key
        order = np.argsort(gk, kind="stable")
        gk, Ur, y = gk[order], Ur[order], ys[order]
        starts = group_starts(gk)
        sizes = np.diff(np.r_[starts, len(gk)])
        big = sizes >= r
        # summed statistics for big groups
        rows_big = np.repeat(big, sizes)
        if big.any():
            sb = starts[big]
            Xb, yb, kb = Ur[rows_big], y[rows_big], gk[rows_big]
            XtX, Xty, n = group_normal_eq(Xb, yb, group_starts(kb))
            s_keys = gk[sb]
        else:
            XtX = np.zeros((0, r, r)); Xty = np.zeros((0, r)); n = np.zeros(0, dtype=np.int64)
            s_keys = np.zeros(0, dtype=np.int64)
        r_keys, r_U, r_y = gk[~rows_big], Ur[~rows_big], y[~rows_big]
        out = {}
        for dq, (k, a, b, c) in split((s_keys // C) % P, s_keys, XtX, Xty, n):
            out.setdefault(dq, [None, None])[0] = (k, a, b, c)
        for dq, (k, a, b) in split((r_keys // C) % P, r_keys, r_U, r_y):
            out.setdefault(dq, [None, None])[1] = (k, a, b)
        for dq, (s_part, r_part) in out.items():
            yield dq, (s_part, r_part)

    return both.mapPartitions(run)


def item_pass(v_round, uploads, ident, reg_param: float, rank: int, num_partitions: int,
              num_clients: int, average: bool):
    """The upload shuffle, then per item partition: all (client, item) normal equations summed,
    solved in one batch, and (average=True) averaged by item weighted by rating counts into new V
    rows; items without any update keep their round-start row. average=False returns the
    client-local versions keyed item * C + client (for the next local epoch)."""
    group_normal_eq, solve_batch, group_starts = _make_kernels(reg_param, rank)
    P, C = int(num_partitions), int(num_clients)
    up = uploads.partitionBy(P, ident)
    both = _check_aligned(v_round.mapValues(lambda x: (0, x)).union(up.mapValues(lambda x: (1, x))),
                          P, "item_pass")

    def run(it):
        import numpy as np
        q = None; vk = vv = None; S, R = [], []
        for q_, (tag, x) in it:
            q = q_
            if tag == 0:
                vk, vv = x
            else:
                if x[0] is not None: S.append(x[0])
                if x[1] is not None: R.append(x[1])
        if vk is None:
            return
        keys_l, XtX_l, Xty_l, n_l = [], [], [], []
        if S:
            keys_l.append(np.concatenate([s[0] for s in S])); XtX_l.append(np.concatenate([s[1] for s in S]))
            Xty_l.append(np.concatenate([s[2] for s in S])); n_l.append(np.concatenate([s[3] for s in S]))
        if R:
            rk = np.concatenate([x[0] for x in R]); rU = np.vstack([x[1] for x in R]); ry = np.concatenate([x[2] for x in R])
            o = np.argsort(rk, kind="stable"); rk, rU, ry = rk[o], rU[o], ry[o]
            st = group_starts(rk)
            a, b, c = group_normal_eq(rU, ry, st)
            keys_l.append(rk[st]); XtX_l.append(a); Xty_l.append(b); n_l.append(c)
        if not keys_l:
            yield q, (vk, vv); return
        keys = np.concatenate(keys_l); XtX = np.concatenate(XtX_l); Xty = np.concatenate(Xty_l)
        n = np.concatenate(n_l)
        o = np.argsort(keys, kind="stable"); keys, XtX, Xty, n = keys[o], XtX[o], Xty[o], n[o]
        st = group_starts(keys)
        keys = keys[st]
        XtX = np.add.reduceat(XtX, st, axis=0); Xty = np.add.reduceat(Xty, st, axis=0)
        n = np.add.reduceat(n, st)
        vc = solve_batch(XtX, Xty, n)                       # one vector per (client, item)
        if not average:
            yield q, (keys, vc); return
        item = keys // C
        ist = group_starts(item)
        num = np.add.reduceat(vc * n[:, None], ist, axis=0)
        den = np.add.reduceat(n, ist)
        new_ids = item[ist]
        out = vv.copy()
        out[np.searchsorted(vk, new_ids)] = num / den[:, None]
        yield q, (vk, out)

    return both.mapPartitions(run, preservesPartitioning=True)


def solve_users(blocks, rows, reg_param: float, rank: int, num_partitions: int):
    """(user_idx, u as list) against the delivered global V. Evaluation only, never timed."""
    group_normal_eq, solve_batch, _ = _make_kernels(reg_param, rank)
    receive = _make_receive()
    solve_users_block = _make_user_solver()
    both = _check_aligned(blocks.mapValues(lambda b: (0, b)).union(rows.mapValues(lambda x: (1, x))),
                          int(num_partitions), "solve_users")

    def run(it):
        block, keys, vrows = receive(it)
        if block is None:
            return
        U, _, _ = solve_users_block(block, keys, vrows, False, 1, group_normal_eq, solve_batch)
        for user, u in zip(block[0].tolist(), U.tolist()):
            yield int(user), u

    return both.mapPartitions(run)
