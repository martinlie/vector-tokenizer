import jax
import jax.numpy as jnp
from jax import random, lax
import optax

def encode_with_channels(x_bins: jnp.ndarray, n_channels: int):
    """
    x_bins: (N, D) integer bins starting at 0
    returns: (N * (2 + 2*D),) flat token stream
    """
    N, D = x_bins.shape
    assert D == n_channels

    BOS = 0
    EOS = 1
    CH_OFFSET = 2
    DATA_OFFSET = 2 + n_channels

    # channel tokens [2 .. 2+D-1]
    ch_tokens = jnp.arange(D, dtype=x_bins.dtype) + CH_OFFSET
    ch_tokens = jnp.broadcast_to(ch_tokens, (N, D))

    # data tokens
    data_tokens = x_bins + DATA_OFFSET

    # interleave CH, DATA
    interleaved = jnp.stack([ch_tokens, data_tokens], axis=2)
    interleaved = interleaved.reshape(N, -1)

    bos = jnp.full((N, 1), BOS, dtype=x_bins.dtype)
    eos = jnp.full((N, 1), EOS, dtype=x_bins.dtype)

    packed = jnp.concatenate([bos, interleaved, eos], axis=1)
    return packed.reshape(-1)

def encode_with_channels_sparse(x_bins: jnp.ndarray, n_channels: int, zero_bin: int):
    """
    Sparse delta encoding (JAX-safe).

    x_bins: (N, D) integer delta bins, 0 = no change
    returns: (variable length,) flat token stream (PAD removed)
    """
    N, D = x_bins.shape
    assert D == n_channels

    BOS = 0
    EOS = 1
    CH_OFFSET = 2
    DATA_OFFSET = 2 + n_channels
    PAD = -1

    # channel ids [0..D-1]
    ch_ids = jnp.arange(D, dtype=x_bins.dtype)

    def encode_row(row_bins):
        # mask non-zero deltas
        mask = row_bins != zero_bin

        # CH / DATA tokens (or PAD)
        ch_tokens = jnp.where(
            mask,
            ch_ids + CH_OFFSET,
            PAD
        )

        data_tokens = jnp.where(
            mask,
            row_bins + DATA_OFFSET,
            PAD
        )

        # interleave CH, DATA (fixed size!)
        interleaved = jnp.stack([ch_tokens, data_tokens], axis=1).reshape(-1)

        # add BOS / EOS
        tokens = jnp.concatenate([
            jnp.array([BOS], dtype=x_bins.dtype),
            interleaved,
            jnp.array([EOS], dtype=x_bins.dtype),
        ])

        return tokens

    # (N, fixed_len)
    rows = jax.vmap(encode_row)(x_bins)

    # flatten and REMOVE PAD tokens
    flat = rows.reshape(-1)
    flat = flat[flat != PAD]

    return flat

def decode_with_channels(flat, n_channels):
    """
    A training block must start at BOS and must not cross an EOS.
    flat: (N*(2+2D),)
    returns: (N, D) bin matrix
    """
    DATA_OFFSET = 2 + n_channels
    row_len = 2 + 2 * n_channels

    rows = flat.reshape(-1, row_len)

    # extract DATA tokens (positions 2,4,6,...)
    data = rows[:, 2:-1:2]

    return data - DATA_OFFSET

#def get_token_batch(token_ids, rng_key, batch_size, n_channels):
#    BOS = 0
#    EOS = 1

#    ROW_LEN = 2 + 2 * n_channels
#    T = token_ids.shape[0]
#    n_rows = T // ROW_LEN

#    rows = token_ids.reshape(n_rows, ROW_LEN)

    # DEBUG ONLY — remove after verification
    # assert bool(jnp.all(rows[:, 0] == BOS))
    # assert bool(jnp.all(rows[:, -1] == EOS))

#    rng_key, subkey = random.split(rng_key)
#    row_idx = random.choice(subkey, n_rows, shape=(batch_size,), replace=False)

#    batch = rows[row_idx]
#    x = batch[:, :-1]
#    y = batch[:, 1:]

#    return x, y

def get_token_batch(
    token_ids: jnp.ndarray,
    rng_key,
    batch_size: int,
    n_channels: int,
    block_size: int,
):
    """
    token_ids: (T,)
    returns:
        x, y: (batch_size, block_size)
    """
    BOS = 0
    ROW_LEN = 2 + 2 * n_channels

    T = token_ids.shape[0]

    # BOS positions are row starts
    n_rows = T // ROW_LEN
    bos_positions = jnp.arange(n_rows) * ROW_LEN

    # keep only BOS where full block fits
    valid = bos_positions[bos_positions + block_size + 1 <= T]

    if valid.shape[0] < batch_size:
        raise ValueError("Not enough BOS-aligned blocks")

    rng_key, subkey = random.split(rng_key)
    starts = random.choice(subkey, valid, shape=(batch_size,), replace=False)

    def slice_block(start):
        seq = lax.dynamic_slice(
            token_ids,
            (start,),
            (block_size + 1,)
        )
        return seq[:-1], seq[1:]

    x, y = jax.vmap(slice_block)(starts)
    return x, y

"""
Loss function with weighted tokens.
* Grammar tokens still produce gradients → model learns sequence validity.
* DATA tokens dominate the loss → model allocates capacity to values.
* Normalization prevents loss scale from drifting when token mix changes.
"""
def loss_fn(
    params,
    apply_fn,
    x,
    token_types,
    channel_ids,
    n_channels,
    y,
    w_data=1.0,
    w_grammar=0.1,
):
    logits = apply_fn(params, x, token_types, channel_ids)

    # per-token CE
    ce = optax.softmax_cross_entropy_with_integer_labels(
        logits, y
    )

    DATA_OFFSET = 2 + n_channels

    # build weights
    weights = jnp.where(
        y >= DATA_OFFSET,
        w_data,       # DATA tokens
        w_grammar,    # BOS / EOS / CH tokens
    )

    # normalize so scale of loss is stable
    loss = (ce * weights).sum() / weights.sum()
    return loss

def compute_token_types(tokens, n_channels):
    """
    tokens: (B, T)
    returns: (B, T) token type ids
    """
    # token ids
    BOS = 0
    EOS = 1
    CH_OFFSET = 2
    DATA_OFFSET = 2 + n_channels

    # token types
    TYPE_BOS = 0
    TYPE_EOS = 1
    TYPE_CH  = 2
    TYPE_DATA = 3
    N_TOKEN_TYPES = 4

    return jnp.where(
        tokens == BOS, TYPE_BOS,
        jnp.where(
            tokens == EOS, TYPE_EOS,
            jnp.where(
                tokens < DATA_OFFSET, TYPE_CH,
                TYPE_DATA
            )
        )
    )

NO_CHANNEL = -1   # sentinel for non-DATA tokens

def compute_channel_ids(tokens, n_channels):
    """
    tokens: (B, T)
    returns: (B, T) with channel index [0..n_channels-1] or -1
    """
    DATA_OFFSET = 2 + n_channels

    # initialize with NO_CHANNEL
    channel_ids = jnp.full(tokens.shape, NO_CHANNEL)

    # DATA tokens are always immediately after a CH token
    # pattern: [CH_k, DATA]
    prev_tokens = jnp.roll(tokens, shift=1, axis=1)

    is_data = tokens >= DATA_OFFSET
    is_channel = (prev_tokens >= 2) & (prev_tokens < DATA_OFFSET)

    # channel index = CH token id - CH_OFFSET
    channel_ids = jnp.where(
        is_data & is_channel,
        prev_tokens - 2,
        channel_ids
    )

    return channel_ids