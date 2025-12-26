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

def fit_quantile_bins_global(data: np.ndarray, n_bins: int):
    """
    Global quantile discretization.

    data: (N, D)
    returns:
        edges: (n_bins-1,)
        mids:  (n_bins,)
    """
    flat = data.reshape(-1)

    # handle degenerate case
    if np.unique(flat).size <= 1:
        edges = np.array([])
        mids = np.array([flat[0]])
        return edges, mids

    qs = np.linspace(0.0, 1.0, n_bins + 1)
    q = np.quantile(flat, qs)

    edges = q[1:-1]                       # (n_bins-1,)
    mids = (q[:-1] + q[1:]) / 2           # (n_bins,)

    return edges, mids

def encode_quantile_global(x, edges):
    """
    Global quantile encoding.

    x:     (T, D)
    edges: (n_bins-1,)
    returns: (T, D) int32 bin indices
    """
    edges = jnp.asarray(edges)
    return jnp.digitize(x, edges)

def decode_quantile_global(tokens, mids):
    """
    Global quantile decoding.

    tokens: (T, D)
    mids:   (n_bins,)
    returns: (T, D) approx continuous values
    """
    mids = jnp.asarray(mids)
    tokens = jnp.clip(tokens, 0, mids.shape[0] - 1)
    return mids[tokens]

def fit_equal_width_bins_global(data: np.ndarray, n_bins: int):
    """
    Global equal-width discretization.

    data: (N, D)
    returns:
        edges: (n_bins-1,)
        mids:  (n_bins,)
    """
    flat = data.reshape(-1)

    lo = flat.min()
    hi = flat.max()

    if lo == hi:
        edges = np.array([])
        mids = np.array([lo])
        return edges, mids

    edges = np.linspace(lo, hi, n_bins + 1)[1:-1]
    mids = np.linspace(lo, hi, n_bins + 1)
    mids = (mids[:-1] + mids[1:]) / 2

    return edges.astype(np.float32), mids.astype(np.float32)

def encode_equal_width_global(x, edges):
    """
    Global equal-width encoding.

    x:     (T, D)
    edges: (n_bins-1,)
    returns: (T, D) int32
    """
    edges = jnp.asarray(edges)
    return jnp.digitize(x, edges)

def decode_equal_width_global(tokens, mids):
    """
    Global equal-width decoding.

    tokens: (T, D)
    mids:   (n_bins,)
    returns: (T, D) approx continuous values
    """
    mids = jnp.asarray(mids)
    tokens = jnp.asarray(tokens)

    tokens_clipped = jnp.clip(tokens, 0, mids.shape[0] - 1)
    values = mids[tokens_clipped]

    return jnp.where(tokens == 0, 0.0, values) # jnp.nan

# u_max = asinh(x_max / s). Controls tail coverage.
# Example: cover x_max=6 stdev with s=1 ⇒ u_max = asinh(6) ≈ 2.49.

def make_asinh_bins_with_zero(n_bins: int, u_max: float):
    """
    n_bins MUST be odd
    """
    assert n_bins % 2 == 1, "n_bins must be odd to have a zero bin"

    half = n_bins // 2

    mids_u_pos = jnp.linspace(0.0, u_max, half + 1)[1:]  # skip zero
    mids_u = jnp.concatenate([-mids_u_pos[::-1], jnp.array([0.0]), mids_u_pos])

    edges_u = (mids_u[:-1] + mids_u[1:]) / 2

    zero_bin = half

    return edges_u, mids_u, zero_bin

def encode_asinh_global(x, edges_u, s: float):
    u = jnp.arcsinh(x / s)
    return jnp.digitize(u, jnp.asarray(edges_u))

def decode_asinh_global(tokens, mids_u, s: float):
    mids_u = jnp.asarray(mids_u)
    tok = jnp.clip(jnp.asarray(tokens), 0, mids_u.shape[0] - 1)
    u_mid = mids_u[tok]
    return s * jnp.sinh(u_mid)