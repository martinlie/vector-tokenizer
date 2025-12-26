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
        masked_logits = jnp.where(mask, logits, -jnp.inf)

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

def preprocess_covariates_from_future_matrix(
    future_tokens,          # (H, n_channels) int32 DATA-tokens
    covariate_cols,         # e.g. [1, 2] meaning columns/channels 1 and 2
    zero_bin: int,          # token meaning "no change / no update"
):
    """
    Build cov_present/cov_value from a dense future token matrix.

    Semantics:
      - If future_tokens[t, ch] == zero_bin, treat as "channel not mentioned" in cov frame t.
      - Otherwise channel is present and the DATA token is forced to that value.

    Returns:
      cov_present: (H, n_channels) bool
      cov_value:   (H, n_channels) int32 (only meaningful where cov_present True)
      covariate_channels: (K,) int32 channel ids
    """
    F = np.asarray(future_tokens)
    if F.ndim != 2:
        raise ValueError(f"future_tokens must be 2D (H, n_channels). Got {F.shape}")

    H, n_channels = F.shape
    covariate_channels = np.asarray(covariate_cols, dtype=np.int32)
    if covariate_channels.ndim != 1:
        raise ValueError("covariate_cols must be 1D")
    if np.any(covariate_channels < 0) or np.any(covariate_channels >= n_channels):
        raise ValueError("covariate_cols contains out-of-range channel index")

    cov_present = np.zeros((H, n_channels), dtype=bool)
    cov_value   = np.zeros((H, n_channels), dtype=np.int32)

    # Extract only covariate columns
    cov_vals = F[:, covariate_channels].astype(np.int32)          # (H, K)
    present  = cov_vals != np.int32(zero_bin)                     # (H, K)

    cov_present[:, covariate_channels] = present
    cov_value[:, covariate_channels]   = cov_vals                 # forced token (even if zero_bin; gated by present)

    return cov_present, cov_value, covariate_channels

# Deterministic ordered frame emission with covariate overwrite/suppression

from functools import partial
import jax
import jax.numpy as jnp
from jax import lax

from functools import partial
import jax
import jax.numpy as jnp
from jax import lax


@partial(
    jax.jit,
    static_argnames=("forward_fn", "vocab_size", "block_size", "n_channels", "n_frames", "zero_bin"),
)
def generate_covariate_frames(
    variables,
    forward_fn,
    token_stream,               # (T,)
    rng_key,
    vocab_size,
    block_size,
    n_channels,
    cov_value_tokens,           # (H, n_channels) DATA-TOKEN ids (i.e. bins + DATA_OFFSET)
    covariate_channels,         # (K,) channel ids, e.g. [1,2]
    n_frames,                   # number of future frames to generate (must be static)
    zero_bin,                   # BIN id, e.g. 2058
    BOS=0,
    EOS=1,
):
    token_stream = jnp.asarray(token_stream, dtype=jnp.int32)
    cov_value_tokens = jnp.asarray(cov_value_tokens, dtype=jnp.int32)
    covariate_channels = jnp.sort(jnp.asarray(covariate_channels, dtype=jnp.int32))

    DATA_OFFSET = 2 + n_channels
    CH_OFFSET = 2
    zero_tok = jnp.int32(DATA_OFFSET + zero_bin)

    # Fixed maximum tokens per frame if we emit all non-cov channels as CH+DATA:
    # BOS + (n_channels * 2) + EOS
    max_tpf = 2 + 2 * n_channels
    out_len = n_frames * max_tpf

    # initial context
    pad_len = max(0, block_size - token_stream.shape[0])
    ctx0 = jnp.concatenate([jnp.zeros((pad_len,), jnp.int32), token_stream[-block_size:]], axis=0)

    emitted_any0 = emitted_any_from_stream(token_stream, n_channels)

    # cov mask
    cov_mask = jnp.zeros((n_channels,), dtype=jnp.bool_).at[covariate_channels].set(True)

    def model_pick_data(ctx, emitted_any, rng):
        # model predicts a DATA token (>= DATA_OFFSET) given current ctx
        ctx_batched = ctx[None, :]
        token_types, channel_ids = infer_token_types_and_channels(ctx_batched, n_channels)

        logits = forward_fn(variables, ctx_batched, token_types, channel_ids)[:, -1]
        last_tok = ctx[-1]
        mask = grammar_mask_from_last(last_tok, emitted_any, vocab_size, n_channels)
        masked_logits = jnp.where(mask, logits, -jnp.inf)

        # deterministic for DATA
        tok = jnp.asarray(jnp.argmax(masked_logits, axis=-1)[0], dtype=jnp.int32)
        return tok, rng

    def write_tok(ctx, tok):
        return jnp.concatenate([ctx[1:], jnp.array([tok], jnp.int32)], axis=0)

    def one_frame(carry, frame_idx):
        ctx, rng, emitted_any = carry

        # We'll build exactly max_tpf tokens, using a small loop and a pointer `p`.
        out = jnp.full((max_tpf,), jnp.int32(EOS))  # default pad
        p = jnp.int32(0)

        def emit(tok, ctx, out, p, emitted_any):
            out = out.at[p].set(tok)
            ctx = write_tok(ctx, tok)
            is_data = tok >= DATA_OFFSET
            emitted_any = lax.select(tok == BOS, False, emitted_any | is_data)
            return ctx, out, p + 1, emitted_any

        # BOS
        ctx, out, p, emitted_any = emit(jnp.int32(BOS), ctx, out, p, emitted_any)

        # channels in canonical order 0..n_channels-1
        def ch_body(ch, state):
            ctx, rng, out, p, emitted_any = state

            # if covariate channel -> force value; if forced value is zero_tok -> suppress channel
            forced = cov_value_tokens[frame_idx, ch]
            is_cov = cov_mask[ch]
            suppress = is_cov & (forced == zero_tok)

            def do_emit(_):
                # emit CH
                ctx2, out2, p2, emitted_any2 = emit(jnp.int32(CH_OFFSET + ch), ctx, out, p, emitted_any)

                # emit DATA
                def cov_data(_):
                    return forced, rng
                def model_data(_):
                    return model_pick_data(ctx2, emitted_any2, rng)

                data_tok, rng2 = lax.cond(is_cov, cov_data, model_data, operand=None)
                ctx3, out3, p3, emitted_any3 = emit(data_tok, ctx2, out2, p2, emitted_any2)
                return (ctx3, rng2, out3, p3, emitted_any3)

            return lax.cond(suppress, lambda _: state, do_emit, operand=None)

        ctx, rng, out, p, emitted_any = lax.fori_loop(
            0, n_channels, ch_body, (ctx, rng, out, p, emitted_any)
        )

        # EOS
        ctx, out, p, emitted_any = emit(jnp.int32(EOS), ctx, out, p, emitted_any)

        return (ctx, rng, emitted_any), out

    (ctxF, rngF, emitted_anyF), frames = lax.scan(one_frame, (ctx0, rng_key, emitted_any0), jnp.arange(n_frames))
    flat = frames.reshape((-1,))
    return flat

@partial(
    jax.jit,
    static_argnames=[
        "forward_fn",
        "vocab_size",
        "block_size",
        "max_new_tokens",
        "n_channels",
        "zero_bin",
    ],
)
def generate_covariate_continue(
    variables,
    forward_fn,
    token_stream,               # (T,)
    rng_key,
    vocab_size,
    block_size,
    max_new_tokens,             # interpreted as a HARD token budget
    n_channels,
    cov_value=None,             # (H, n_channels) DATA tokens (future matrix)
    covariate_channels=None,    # (K,) channel ids, e.g. [2,3]
    zero_bin=0,                 # DATA token meaning "no change" => suppress cov channel
    BOS=0,
    EOS=1,
):
    token_stream = jnp.asarray(token_stream, dtype=jnp.int32)

    if cov_value is None:
        cov_value = jnp.zeros((1, n_channels), dtype=jnp.int32)
    else:
        cov_value = jnp.asarray(cov_value, dtype=jnp.int32)

    if covariate_channels is None:
        covariate_channels = jnp.zeros((0,), dtype=jnp.int32)
    else:
        covariate_channels = jnp.asarray(covariate_channels, dtype=jnp.int32)

    # cov_mask[ch] True if channel is covariate-controlled
    cov_mask = jnp.zeros((n_channels,), dtype=jnp.bool_)
    cov_mask = lax.cond(
        covariate_channels.size > 0,
        lambda m: m.at[jnp.sort(covariate_channels)].set(True),
        lambda m: m,
        cov_mask,
    )

    DATA_OFFSET = 2 + n_channels
    CH_OFFSET = 2
    H = cov_value.shape[0]

    emitted_any0 = emitted_any_from_stream(token_stream, n_channels)

    def cov_tok(frame_idx, ch):
        # beyond provided horizon => treat as zero_bin (suppressed)
        return lax.cond(
            frame_idx < H,
            lambda _: cov_value[frame_idx, ch],
            lambda _: jnp.int32(zero_bin),
            operand=None,
        ).astype(jnp.int32)

    # ---------------------------------------------------------------------
    # We generate a stream by emitting:
    #   BOS,
    #   for ch in 0..n_channels-1:
    #       if cov_mask[ch]: emit CH+DATA only if cov_tok != zero_bin
    #       else: emit CH + (model-predicted DATA)
    #   EOS
    # repeat until token budget exhausted
    #
    # This guarantees:
    # - covariate channels match future matrix when non-zero_bin
    # - covariate channels are suppressed when zero_bin
    # - non-cov channels are predicted by the model
    # - channel order is canonical
    # ---------------------------------------------------------------------

    def step(carry, _):
        ctx, rng, emitted_any, frame_idx, phase, ch, sub = carry
        # phase: 0 = emit BOS, 1 = emit channels, 2 = emit EOS
        # sub (only in phase=1): 0 = emit CH, 1 = emit DATA

        def emit_bos(_):
            return jnp.int32(BOS), jnp.int32(1), jnp.int32(0), jnp.int32(0), rng

        def emit_eos(_):
            return jnp.int32(EOS), jnp.int32(0), jnp.int32(0), jnp.int32(0), rng

        def emit_channel(_):
            # Skip covariate channels whose cov token is zero_bin (emit nothing, just advance ch)
            def skip_cond(state):
                ch_ = state
                v = cov_tok(frame_idx, ch_)
                return (ch_ < n_channels) & cov_mask[ch_] & (v == jnp.int32(zero_bin))

            def skip_body(ch_):
                return ch_ + 1

            ch2 = lax.while_loop(skip_cond, skip_body, ch)

            # If we've finished channels, next is EOS
            done = ch2 >= n_channels

            def when_done(_):
                # return a dummy token; caller will switch phase to EOS next
                return jnp.int32(-1), jnp.int32(2), jnp.int32(0), jnp.int32(0), rng

            def when_not_done(_):
                is_cov = cov_mask[ch2]

                def emit_ch(_):
                    tok = jnp.int32(CH_OFFSET) + jnp.int32(ch2)
                    return tok, jnp.int32(1), jnp.int32(ch2), jnp.int32(1), rng  # next sub=DATA

                def emit_data(_):
                    # If covariate channel: force cov token
                    def cov_data(_):
                        tok = cov_tok(frame_idx, ch2)
                        return tok.astype(jnp.int32), rng

                    # Else: sample/argmax data token from model, given context after CH
                    def model_data(_):
                        ctx_batched = ctx[None, :]
                        token_types, channel_ids = infer_token_types_and_channels(ctx_batched, n_channels)

                        logits = forward_fn(
                            variables,
                            ctx_batched,
                            token_types,
                            channel_ids,
                        )[:, -1]

                        last_tok = ctx[-1]
                        mask = grammar_mask_from_last(last_tok, emitted_any, vocab_size, n_channels)
                        masked_logits = jnp.where(mask, logits, -jnp.inf)

                        rng2, subkey = jax.random.split(rng)

                        # DATA step is deterministic to reduce noise, mirroring your original preference
                        tok = jnp.asarray(jnp.argmax(masked_logits, axis=-1)[0], dtype=jnp.int32)
                        return tok, rng2

                    tok, rng2 = lax.cond(is_cov, cov_data, model_data, operand=None)

                    # After DATA, advance to next channel
                    return tok, jnp.int32(1), jnp.int32(ch2 + 1), jnp.int32(0), rng2

                return lax.cond(sub == 0, emit_ch, emit_data, operand=None)

            return lax.cond(done, when_done, when_not_done, operand=None)

        # choose emission based on phase
        next_tok, next_phase, next_ch, next_sub, rng = lax.cond(
            phase == 0,
            emit_bos,
            lambda _: lax.cond(phase == 1, emit_channel, emit_eos, operand=None),
            operand=None,
        )

        # If channel phase returned -1 (means channels done), emit EOS next
        phase = lax.select(next_tok == jnp.int32(-1), jnp.int32(2), next_phase)
        next_tok = lax.select(next_tok == jnp.int32(-1), jnp.int32(EOS), next_tok)

        # update emitted_any
        is_data = next_tok >= DATA_OFFSET
        emitted_any = lax.select(next_tok == BOS, False, emitted_any | is_data)

        # frame counter sync: advance on EOS
        frame_idx = frame_idx + (next_tok == EOS).astype(jnp.int32)

        # roll context and append
        ctx = jnp.concatenate([ctx[1:], jnp.array([next_tok], dtype=jnp.int32)], axis=0)
        return (ctx, rng, emitted_any, frame_idx, phase, next_ch, next_sub), next_tok

    # init ctx (pad left)
    pad_len = max(0, block_size - token_stream.shape[0])
    pad = jnp.zeros((pad_len,), dtype=jnp.int32)
    ctx0 = jnp.concatenate([pad, token_stream[-block_size:]], axis=0)

    carry0 = (
        ctx0,
        rng_key,
        emitted_any0,
        jnp.int32(0),   # frame_idx (future row index)
        jnp.int32(0),   # phase (start by emitting BOS)
        jnp.int32(0),   # ch
        jnp.int32(0),   # sub
    )

    (_, _, _, _, _, _, _), new_tokens = lax.scan(step, carry0, None, length=max_new_tokens)
    return new_tokens


def pad_tokens(tokens, T_max=512):
      PAD = 0  # BOS
      return jnp.pad(tokens, (0, T_max - tokens.shape[0]), constant_values=PAD)

from functools import partial
import jax
import jax.numpy as jnp
from jax import lax


@partial(
    jax.jit,
    static_argnames=("n_channels", "zero_bin"),
)
def decode_with_channels_stream(flat, n_channels, zero_bin):
    """
    Decode a flat token stream into completed frames.

    Rules:
    - Frames start at BOS (0)
    - Frames end at EOS (1)
    - CH tokens select channel
    - DATA tokens assign value to last CH
    - Unfinished trailing frames are skipped

    IMPORTANT semantics for delta models:
    - Channels not mentioned in a frame decode to `zero_bin` (no change).
    - Empty frames (BOS ... EOS with no DATA) are still emitted as all-zero_bin rows.
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
                jnp.full((n_channels,), jnp.asarray(zero_bin, dtype=flat.dtype), dtype=flat.dtype),
                jnp.zeros((n_channels,), dtype=jnp.bool_),
                jnp.int32(-1),
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
            lambda _: (tok - CH_OFFSET).astype(jnp.int32),
            lambda _: current_ch,
            operand=None,
        )

        # ---- data write ----
        def write_data_fn(_):
            row = current_row.at[current_ch].set((tok - DATA_OFFSET).astype(flat.dtype))
            s = seen.at[current_ch].set(True)
            return row, s

        current_row, seen = lax.cond(
            frame_active & is_data & (current_ch >= 0),
            write_data_fn,
            lambda _: (current_row, seen),
            operand=None,
        )

        # ---- EOS emits frame ----
        # Emit even if `seen` is all False: empty frame == all zero_bin changes
        emit = frame_active & is_eos

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
        jnp.full((n_channels,), jnp.asarray(zero_bin, dtype=flat.dtype), dtype=flat.dtype),
        jnp.zeros((n_channels,), dtype=jnp.bool_),
        jnp.int32(-1),
        jnp.zeros((max_frames, n_channels), flat.dtype),
        jnp.zeros((max_frames,), dtype=jnp.bool_),
        jnp.int32(0),
    )

    final_carry, _ = lax.scan(step, init_carry, flat)
    out_rows, out_idx = final_carry[4], final_carry[6]
    return out_rows, out_idx