import jax
from jax import random
import jax.numpy as jnp
import optax
from functools import partial
import numpy as np
import pandas as pd

def get_batch(data, rng_key, batch_size, block_size):
    """
    Extracts a random batch of input and target data
    Args:
        data: An array of all the data's token ID's.
        rng_key: Random number generator key.
        batch_size: Number of parallel batches.
        block_size: Maximum time length for the token sequence.
    Returns:
        Input token ID's and target token ID's.
    """
    ix = random.randint(key=rng_key, shape=(batch_size, ), minval=0, maxval=len(data) - block_size)
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


@partial(jax.jit, static_argnames=['forward_fn', 'vocab_size', 'batch_size', 'block_size', 'max_new_tokens'])
def generate(variables, forward_fn, index_seq, rng_key, vocab_size, batch_size, block_size, max_new_tokens):
    """
    Generates max_new_tokens number of new tokens, 
    given the starting sequence of tokens index_seq 
    Args:
        variables: Bigram models parameters.
        forward_fn: Function that performs a forward pass of the model.
        index_seq: Array of token indices with shape (B, T), 
            where B is the batch size and T is the time steps.
        rng_key: Random number generator key.
        vocab_size: Number of independent tokens in the vocabulary.
        batch_size: Number of parallel batches.
        max_new_tokens: Maximum number of new tokens to generate
    Returns:
        An array of generated indices
    """
    # Batched sampling function
    batched_choice = jax.vmap(jax.random.choice)
    
    for _ in range(max_new_tokens):
        # Crop index_seq to the last block_size tokens
        index_cond = index_seq[:, -block_size:]
        logits = forward_fn(variables, index_cond)
        # Focus only on the last time step
        # Shape changes from (B, T, C) -> (B, C)
        logits = logits[:, -1, :]
        # Convert to probabilities using softmax
        probs = jax.nn.softmax(logits, axis=-1)
        # Sample a token index using probs
        rng_key, subkey = jax.random.split(rng_key)
        batched_key = subkey.reshape(1, -1)
        batched_key = jnp.repeat(batched_key, batch_size, axis=0)
        a = jnp.arange(vocab_size).reshape(1, -1)
        a = jnp.repeat(a, batch_size, axis=0)
        next_indexes = batched_choice(batched_key, a, p=probs)
        next_indexes = next_indexes.reshape(batch_size, -1)
        # Append the sampled index to the running sequence
        index_seq = jnp.concatenate([index_seq, next_indexes], axis=1)
    return index_seq


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