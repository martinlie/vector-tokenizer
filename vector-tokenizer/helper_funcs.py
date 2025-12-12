import jax
from jax import random
import jax.numpy as jnp
import optax
from functools import partial
import jax.lax as lax
import numpy as np
import pandas as pd

@partial(
    jax.jit,
    static_argnames=['forward_fn', 'vocab_size', 'batch_size', 'block_size', 'max_new_tokens']
)
def generate(variables, forward_fn, index_seq, rng_key,
             vocab_size, batch_size, block_size, max_new_tokens):

    B, T = index_seq.shape
    final_T = T + max_new_tokens

    # ---------- FIX: Pre-pad with block_size zeros on the left ----------
    pad = jnp.zeros((B, block_size), dtype=index_seq.dtype)
    out = jnp.concatenate([pad, index_seq], axis=1)  # shape (B, block_size + T)

    # We will write new tokens starting from position block_size + T
    def step(carry, t_offset):
        out, rng = carry

        # Slice exactly block_size tokens (static shape)
        start = t_offset                           # dynamic index
        index_cond = lax.dynamic_slice(out, (0, start), (B, block_size))

        # Run model on the block-size window
        logits = forward_fn(variables, index_cond)
        logits = logits[:, -1]                     # (B, vocab)

        # Sample
        rng, sub = jax.random.split(rng)
        next_idx = jax.random.categorical(sub, logits=logits)

        # Write next token at out[:, t_offset + block_size]
        out = lax.dynamic_update_slice(
            out,
            next_idx[:, None],
            (0, t_offset + block_size)
        )

        return (out, rng), None

    # t_offset will run from T ... final_T-1
    (out, _), _ = lax.scan(
        step,
        (out, rng_key),
        jnp.arange(T, final_T)
    )

    # Remove the left padding again
    return out[:, block_size:]

def masked_fill(mask, a, fill):
    return jax.lax.select(mask, a, jax.lax.broadcast(fill, a.shape))


def loss_fn(variables, forward_fn, index_seq, labels):
    """
    Calculates the cross entropy loss of 
    all batches and time steps, 
    then returns the mean.
    Args:
        variables: Language model parameters.
        forward_fn: Function that performs a forward pass of the model.
        index_seq: Array of token indices with shape (B, T), 
            where B is the batch size and T is the time steps.
        labels: Indexes of the next token in the sequence.
    Returns:
        Cross entropy loss
    """
    logits = forward_fn(variables, index_seq)

    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    # Average loss across all batches and time steps
    loss = loss.mean()
    return loss


"""encoder: take a data frame, output a list of integers"""
def encode(df):
    v = df['vector_id'].to_numpy()
    first_nonzero = np.argmax(v != 0)
    vector_ids = v[first_nonzero:].tolist() if np.any(v != 0) else []
    return vector_ids

"""decoder: take a list of integers, output a dataframe"""
def decode(vector_ids, df, vector_col='vector_id'):
    mapping = df[df[vector_col] != 0].set_index(vector_col)
    decoded_rows = []
    last_row = None
    
    for vid in vector_ids:
        if vid == 0:
            # repeat previous row if available
            if last_row is not None:
                decoded_rows.append(last_row)
            else:
                # first entry is zero → append NaNs
                decoded_rows.append(pd.Series({col: None for col in df.columns if col != vector_col}))
        else:
            # fetch row from mapping
            row = mapping.loc[vid]
            # If mapping has multiple rows per vector_id, take the first
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            decoded_rows.append(row)
            last_row = row
            
    # Combine into a DataFrame  
    decoded_df = pd.DataFrame(decoded_rows).reset_index(drop=True)
    return decoded_df

def get_vector_batch(vector_ids, rng_key, batch_size, block_size):
    v = np.array(vector_ids)  # ensure numpy array
    n = len(v)
    
    # find all valid start indices (where a full block fits and first vector_id != 0)
    valid_starts = np.where(v[:n-block_size] != 0)[0]
    
    if len(valid_starts) < batch_size:
        raise ValueError("Not enough valid sequences to sample the batch.")
    
    # randomly choose start indices
    idx = random.choice(rng_key, jnp.array(valid_starts), shape=(batch_size,), replace=False)
    
    x_batch = []
    y_batch = []
    
    for i in idx:
        seq = v[i:i+block_size]
        target = v[i+1:i+block_size+1]
        x_batch.append(seq)
        y_batch.append(target)
    
    x = jnp.stack(x_batch)
    y = jnp.stack(y_batch)
    
    return x, y