import os
import requests
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from jax import value_and_grad
import pickle
import pandas as pd

from helper_funcs import generate, masked_fill, loss_fn
from tqdm import tqdm


class FeedForward(nn.Module):
    """
    A feed forward multi-layer perceptron network.
    """
    n_embed: int
    drop_rate: float

    @nn.compact
    def __call__(self, x):
        net = nn.Sequential([
            nn.Dense(4 * self.n_embed),
            jax.nn.relu,
            nn.Dense(self.n_embed),
            nn.Dropout(rate=self.drop_rate, deterministic=True)
        ])
        x = net(x)

        return x

class Head(nn.Module):
    """
    A single-headed self-attention decoder block.
    Takes the combined token and position embedding as input,
    then calculates the key and query values.
    The key and query are multiplied to calculate the 
    attention scores/affinities. The future weights are
    then altered to have zero affinity, this ensures the 
    model can't "cheat". The input is then used to calculate
    the values, which are then aggregated by multiplying 
    them with the weights.
    """
    head_size: int
    drop_rate: float

    @nn.compact
    def __call__(self, x):
        B,T,C = x.shape
        key = nn.Dense(self.head_size, use_bias=False)
        k = key(x) # (B,T,C)
        query = nn.Dense(self.head_size, use_bias=False)
        q = query(x) # (B,T,C)
        # compute attention scores ("affinities")
        weights =  q @ k.transpose((0, -1, -2)) * self.head_size**-0.5 # (B, T, C) @ (B, C, T) ---> (B, T, T)
        tril = jnp.tril(jnp.ones(shape=(T, T), dtype=bool))
        tril = jnp.repeat(tril[None, ...], repeats=B, axis=0)
        weights = masked_fill(tril, weights, -jnp.inf)
        weights = jax.nn.softmax(weights, axis=-1)
        drop = nn.Dropout(rate=self.drop_rate, deterministic=True)
        weights = drop(weights)
        # perform the weighted aggregation of the values
        value = nn.Dense(self.head_size, use_bias=False)
        v = value(x)
        out = weights @ v
        return out

class MultiHeadedAttention(nn.Module):
    """
    Combines multiple heads of scaled self-attention 
    in parallel, then concatenates the heads outputs.
    """
    num_heads: int
    head_size: int
    n_embed: int
    drop_rate: float

    @nn.compact
    def __call__(self, x):
        # Create a list of num_heads heads
        heads = [Head(self.head_size, self.drop_rate) for _ in range(self.num_heads)]
        # Provide the same input for each head
        heads_out = [h(x) for h in heads]
        combined_logits = jnp.concatenate(heads_out, axis=-1)
        # Perform a linear projection of the self-attention
        proj = nn.Dense(self.n_embed)
        logits = proj(combined_logits)
        drop = nn.Dropout(rate=self.drop_rate, deterministic=True)
        logits = drop(logits)
        return logits

class Block(nn.Module):
    """
    Transformer decoder block.
    It combines communication and computation.
    The communication is performed by the 
    multi-headed attention layer.
    Then the computation is performed by 
    the feed forward block.
    Skip connections are used to make the block scalable 
    and layer norm is used to speed up training.
    """
    n_embed: int
    num_heads: int
    drop_rate: float

    @nn.compact
    def __call__(self, x):
        head_size = self.n_embed // self.num_heads
        sa_heads = MultiHeadedAttention(self.num_heads, head_size, self.n_embed, self.drop_rate)
        # Using skip connections with x + heads
        x = x + sa_heads(nn.LayerNorm()(x)) # apply one head of self-attention (B, T, C)
        ffwd = FeedForward(self.n_embed, self.drop_rate)
        x = x + ffwd(nn.LayerNorm()(x))
        return x

class GPT2(nn.Module):
    """
    GPT-2 language model.
    Uses the previous tokens in the sequence to 
    determine the probabilities of the next token.
    Processes the combined position and token embedding
    through multiple transformer decoder blocks, 
    which is then processed through a dense layer to 
    aquire the token logits.
    The logits can then be processed through a softmax
    function to calculate the token probabilities.
    """
    vocab_size: int
    n_embed: int
    block_size: int
    num_heads: int
    num_layers: int
    drop_rate: float
    
    @nn.compact
    def __call__(self, index_seq):
        B, T = index_seq.shape

        # Each token directly reads off the logits for the next token from a lookup table
        token_embedding_table = nn.Embed(num_embeddings=self.vocab_size, features=self.n_embed) 
        token_emb = token_embedding_table(index_seq) # (B, T, C)

        position_embedding_table = nn.Embed(num_embeddings=self.block_size, features=self.n_embed) 
        pos_emb = position_embedding_table(jnp.arange(T)) # (T, C)

        x = token_emb + pos_emb # (B, T, C)

        decoder_blocks = [Block(self.n_embed, num_heads=self.num_heads, drop_rate=self.drop_rate) for _ in range(self.num_layers)]
        decoder_blocks.append(nn.LayerNorm())
        blocks = nn.Sequential(
            decoder_blocks
        )
        x = blocks(x)

        lm_head = nn.Dense(self.vocab_size)
        logits = lm_head(x) # (B, T, vocab_size)

        return logits

