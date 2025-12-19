import numpy as np
import jax.numpy as jnp
from jax import vmap

## Example
## original data
# X = df.values.astype(np.float32)   # (N, D)
## fit bins
 #edges, mids = fit_quantile_bins(X, n_bins=8)
## encode
#X_tok = encode_quantile(jnp.asarray(X), edges)
## decode
#X_rec = decode_quantile(X_tok, mids)

def fit_quantile_bins(data: np.ndarray, n_bins: int):
    """
    data: (N, D)
    returns:
        edges: (D, n_bins-1)
        mids:  (D, n_bins)
    """
    D = data.shape[1]

    edges = []
    mids = []

    qs = np.linspace(0, 1, n_bins + 1)

    for d in range(D):
        col = data[:, d]

        if np.unique(col).size <= 1:
            # constant column
            e = np.array([])
            m = np.array([col[0]])
        else:
            q = np.quantile(col, qs)
            e = q[1:-1]
            m = (q[:-1] + q[1:]) / 2

        edges.append(e)
        mids.append(m)

    return edges, mids

def encode_quantile(x, edges):
    """
    x:     (T, D)
    edges: list[D] of (n_bins-1,)
    returns tokens: (T, D) int32
    """
    def encode_col(col, e):
        return jnp.digitize(col, e)

    return jnp.stack([
        encode_col(x[:, d], jnp.asarray(edges[d]))
        for d in range(x.shape[1])
    ], axis=1)

def decode_quantile(tokens, mids):
    """
    tokens: (T, D)
    mids:   list[D] of (n_bins,)
    returns approx continuous values (T, D)
    """
    def decode_col(tok, m):
        m = jnp.asarray(m)
        tok = jnp.clip(tok, 0, m.shape[0] - 1)
        return m[tok]

    return jnp.stack([
        decode_col(tokens[:, d], mids[d])
        for d in range(tokens.shape[1])
    ], axis=1)

## Example
#X = df.values.astype(np.float32)
#edges, mids = fit_equal_width_bins(X, n_bins=8)
#X_tok = encode_equal_width(jnp.asarray(X), edges)
#X_rec = decode_equal_width(X_tok, mids)


def fit_equal_width_bins(data: np.ndarray, n_bins: int | list[int]):
    """
    data: (N, D)
    n_bins: int or list[int] per column

    returns:
        edges: list[D] of (bins-1,)
        mids:  list[D] of (bins,)
    """
    D = data.shape[1]
    bins = n_bins if isinstance(n_bins, list) else [n_bins] * D

    edges = []
    mids = []

    for d in range(D):
        col = data[:, d]
        b = bins[d]

        lo, hi = col.min(), col.max()

        if lo == hi:
            e = np.array([])
            m = np.array([lo])
        else:
            e = np.linspace(lo, hi, b + 1)[1:-1]
            m = np.linspace(lo, hi, b + 1)
            m = (m[:-1] + m[1:]) / 2

        edges.append(e.astype(np.float32))
        mids.append(m.astype(np.float32))

    return edges, mids

def encode_equal_width(x, edges):
    """
    x:     (T, D)
    edges: list[D] of (bins-1,)
    returns tokens: (T, D) int32
    """
    return jnp.stack([
        jnp.digitize(x[:, d], jnp.asarray(edges[d]))
        for d in range(x.shape[1])
    ], axis=1)

def decode_equal_width(tokens, mids):
    """
    tokens: (T, D)
    mids:   list[D] of (bins,)
    returns approx continuous values (T, D)
    """
    return jnp.stack([
        jnp.asarray(mids[d])[jnp.clip(tokens[:, d], 0, len(mids[d]) - 1)]
        for d in range(tokens.shape[1])
    ], axis=1)

