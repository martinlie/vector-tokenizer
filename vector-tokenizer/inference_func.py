import os
import requests
import numpy as np
import jax, jaxlib
import jax.numpy as jnp
import flax.linen as nn
import optax
from jax import value_and_grad, random, lax
import pickle
import pandas as pd
import math
from functools import partial
from datetime import datetime

from helper_funcs import generate, masked_fill
from tqdm import tqdm
from attention_model import *

def infer_token_types_and_channels(tokens, n_channels):
    """
    tokens: (B, T)
    returns:
        token_types: (B, T)
        channel_ids: (B, T)
    """
    BOS = 0
    EOS = 1
    CH_OFFSET = 2
    DATA_OFFSET = 2 + n_channels

    # token type ids
    token_types = jnp.where(
        tokens == BOS, 0,
        jnp.where(
            tokens == EOS, 1,
            jnp.where(
                tokens < DATA_OFFSET, 2,  # CH
                3                           # DATA
            )
        )
    )

    # channel ids
    channel_ids = jnp.where(
        (tokens >= CH_OFFSET) & (tokens < DATA_OFFSET),
        tokens - CH_OFFSET,
        0
    )

    return token_types, channel_ids

def token_type(tok, n_channels):
    BOS = 0
    EOS = 1
    DATA_OFFSET = 2 + n_channels

    return jnp.where(
        tok == BOS, 0,
        jnp.where(
            tok == EOS, 1,
            jnp.where(tok < DATA_OFFSET, 2, 3)
        )
    )

def grammar_mask_from_last(
    last_tok,
    emitted_any,
    vocab_size,
    n_channels,
):
    BOS = 0
    EOS = 1
    CH_OFFSET = 2
    DATA_OFFSET = 2 + n_channels

    ttype = token_type(last_tok, n_channels)
    mask = jnp.zeros((vocab_size,), dtype=jnp.bool_)

    def bos_case(m):
        return m.at[CH_OFFSET:DATA_OFFSET].set(True)

    def ch_case(m):
        return m.at[DATA_OFFSET:].set(True)

    def data_case(m):
        m = m.at[CH_OFFSET:DATA_OFFSET].set(True)
        m = m.at[EOS].set(emitted_any)
        return m

    def eos_case(m):
        return m.at[BOS].set(True)

    return lax.switch(
        ttype,
        [bos_case, eos_case, ch_case, data_case],
        mask,
    )

def emitted_any_from_stream(stream, n_channels):
    BOS = 0
    DATA_OFFSET = 2 + n_channels

    T = stream.shape[0]
    idx = jnp.arange(T)

    # index of last BOS
    is_bos = stream == BOS
    last_bos = jnp.max(jnp.where(is_bos, idx, -1))

    # DATA tokens strictly after last BOS
    is_data_after = (stream >= DATA_OFFSET) & (idx > last_bos)

    return jnp.any(is_data_after)

@partial(
    jax.jit,
    static_argnames=[
        'forward_fn',
        'vocab_size',
        'block_size',
        'max_new_tokens',
        'n_channels',
    ],
)
def generate_continue(
    variables,
    forward_fn,
    token_stream,     # (T,)
    rng_key,
    vocab_size,
    block_size,
    max_new_tokens,
    n_channels,
    BOS = 0,
    EOS=1,
):
    token_stream = jnp.asarray(token_stream)

    # infer emitted_any from history
    DATA_OFFSET = 2 + n_channels
    emitted_any = emitted_any_from_stream(token_stream, n_channels)

    def step(carry, _):
        ctx, rng, emitted_any = carry
        ctx_batched = ctx[None, :]  # (1, T)

        token_types, channel_ids = infer_token_types_and_channels(
            ctx_batched, n_channels # was ctx
        )

        logits = forward_fn(
            variables,
            ctx_batched, #ctx,
            token_types,
            channel_ids,
        )[:, -1]  # (1, vocab)

        last_tok = ctx[-1] #ctx[0, -1]

        mask = grammar_mask_from_last(
            last_tok,
            emitted_any,
            vocab_size,
            n_channels,
        )

        rng, sub = jax.random.split(rng)
        masked_logits = jnp.where(mask, logits, -jnp.inf) #-1e9)
        #next_tok = jax.random.categorical(sub, masked_logits)[0]

        # prefer deterministic (greedy) selection for DATA tokens to reduce randomness;
        # sample stochastically for other token types
        ttype = token_type(last_tok, n_channels)
        def pick_data(_):
            # argmax over last axis gives a (1,) array; take [0] to return a scalar
            return jnp.asarray(jnp.argmax(masked_logits, axis=-1)[0], dtype=jnp.int32)
        def pick_other(_):
            # categorical returns shape (1,), index [0] -> scalar
            return jax.random.categorical(sub, masked_logits)[0]
        next_tok = lax.cond(ttype == 2, pick_data, pick_other, operand=None)

        # update emitted_any
        is_data = next_tok >= DATA_OFFSET
        emitted_any = lax.select(
            next_tok == BOS, 
            False, 
            emitted_any | is_data
        )

        # roll context left and append next token
        ctx = jnp.concatenate([ctx[1:], jnp.array([next_tok])], axis=0)
        return (ctx, rng, emitted_any), next_tok

    # pad on the left if token_stream is shorter than block_size
    pad = jnp.zeros((max(0, block_size - token_stream.shape[0]),), dtype=token_stream.dtype)
    ctx0 = jnp.concatenate([pad, token_stream[-block_size:]], axis=0)

    (_, _, _), new_tokens = lax.scan(
        step,
        (ctx0, rng_key, emitted_any),
        None,
        length=max_new_tokens,
    )

    return new_tokens

def pad_tokens(tokens, T_max=512):
      PAD = 0  # BOS
      return jnp.pad(tokens, (0, T_max - tokens.shape[0]), constant_values=PAD)

@partial(
      jax.jit, 
      static_argnames=(
            "n_channels",
      )
)
def decode_with_channels_stream(flat, n_channels):
    """
    Decode a flat token stream into completed frames.

    Rules:
    - Frames start at BOS (0)
    - Frames end at EOS (1)
    - CH tokens select channel
    - DATA tokens assign value to last CH
    - Unfinished trailing frames are skipped

    Returns:
        (N, n_channels) array
    """
    BOS = 0
    EOS = 1
    CH_OFFSET = 2
    DATA_OFFSET = 2 + n_channels

    flat = jnp.asarray(flat)
    L = flat.shape[0]
    max_frames = L // 2 + 1

    def step(carry, tok):
        (
            frame_active,
            current_row,
            seen,
            current_ch,
            out_rows,
            out_mask,
            out_idx,
        ) = carry

        is_bos = tok == BOS
        is_eos = tok == EOS
        is_ch = (tok >= CH_OFFSET) & (tok < DATA_OFFSET)
        is_data = tok >= DATA_OFFSET

        # ---- BOS starts frame ----
        def start_frame_fn(_):
            return (
                True,
                jnp.zeros_like(current_row),
                jnp.zeros_like(seen),
                -1,
            )

        frame_active, current_row, seen, current_ch = lax.cond(
            is_bos & (~frame_active),
            start_frame_fn,
            lambda _: (frame_active, current_row, seen, current_ch),
            operand=None,
        )

        # ---- channel select ----
        current_ch = lax.cond(
            frame_active & is_ch,
            lambda _: tok - CH_OFFSET,
            lambda _: current_ch,
            operand=None,
        )

        # ---- data write ----
        def write_data_fn(_):
            row = current_row.at[current_ch].set(tok - DATA_OFFSET)
            s = seen.at[current_ch].set(True)
            return row, s

        current_row, seen = lax.cond(
            frame_active & is_data & (current_ch >= 0),
            write_data_fn,
            lambda _: (current_row, seen),
            operand=None,
        )

        # ---- EOS emits frame ----
        emit = frame_active & is_eos & jnp.any(seen)

        def emit_fn(_):
            rows = out_rows.at[out_idx].set(current_row)
            mask = out_mask.at[out_idx].set(True)
            return rows, mask, out_idx + 1

        out_rows, out_mask, out_idx = lax.cond(
            emit,
            emit_fn,
            lambda _: (out_rows, out_mask, out_idx),
            operand=None,
        )

        # ---- reset on EOS ----
        frame_active = jnp.where(is_eos, False, frame_active)
        current_ch = jnp.where(is_eos, -1, current_ch)

        return (
            frame_active,
            current_row,
            seen,
            current_ch,
            out_rows,
            out_mask,
            out_idx,
        ), None

    init_carry = (
        False,
        jnp.zeros((n_channels,), flat.dtype),
        jnp.zeros((n_channels,), bool),
        -1,
        jnp.zeros((max_frames, n_channels), flat.dtype),
        jnp.zeros((max_frames,), bool),
        0,
    )

    (final_carry, _) = lax.scan(step, init_carry, flat)

    out_rows, out_mask, out_idx = final_carry[4], final_carry[5], final_carry[6]

    return out_rows, out_idx