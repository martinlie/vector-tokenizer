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
from pathlib import Path

from helper_funcs import generate, masked_fill
from tqdm import tqdm
import matplotlib.pyplot as plt
from attention_model import *

# Disable JIT compilation
#jax.config.update("jax_disable_jit", True)

import discretize_func as discretize
import tokenizer_func as tokenizer

def status():
      print("jax", jax.__version__, "jaxlib", jaxlib.__version__)
      print(jax.default_backend())
      print(jax.devices())

def get_split_data(path: str = "./data", resample_interval: str = "h"): # or "15min"
      # Read
      df = pd.read_parquet(Path(path) / "dehli.parquet")

      # Fill nans with nearest
      df = df.ffill()

      # Downsample from 5-min resolution
      df = df.resample(resample_interval).mean()

      # Fill nans with nearest
      df = df.ffill()

      # Columns of interest
      cols = ['Power demand', 'temp', 'dwpt', 'rhum', 'wdir', 'wspd', 'pres']
      df = df[cols]

      # Split in train/test
      split_idx = int(len(df) * 0.8)
      X = df.iloc[:split_idx].copy()
      Y = df.iloc[split_idx:].copy()

      return X, Y

def make_train_step(apply_fn, optimizer, n_channels):
    @jax.jit
    def train_step(variables, opt_state, xb, yb, token_types, channel_ids):
        # Ensure pure JAX function
        def loss_fn(params):
            return tokenizer.loss_fn(
                params,
                apply_fn,
                xb,
                token_types,
                channel_ids,
                n_channels,
                yb,
            )

        loss, grads = jax.value_and_grad(loss_fn)(variables)
        updates, opt_state = optimizer.update(grads, opt_state, variables)
        variables = optax.apply_updates(variables, updates)

        # JAX-friendly NaN flag (don’t branch inside jit; return it)
        is_nan = jnp.isnan(loss)
        return variables, opt_state, loss, is_nan

    return train_step



def train(model_name, rng_key, epochs, learning_rate, train_tokens, mu, sigma, resample_interval,
            batch_size, n_channels, block_size, n_embed, num_heads, num_layers, drop_rate, 
            vocab_size, n_bins, edges, mids, zero_bin, s, x_max, variables = None, model = None):

      MODEL_DIR = Path("./models")
      MODEL_DIR.mkdir(parents=True, exist_ok=True)

      # Establish optimizer
      learning_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=3e-5,
            warmup_steps=0.05 * epochs,
            decay_steps=epochs,
            end_value=1e-6,
      )
      optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_schedule) # learning_rate
      )

      if model_name is None:
            model_name = MODEL_DIR / f"token_model_{datetime.now():%Y%m%d_%H%M%S}.pkl"
            print("Model:", model_name)

            # Set up and spin model
            rng_key, subkey = jax.random.split(rng_key)
            xb, yb = tokenizer.get_token_batch(train_tokens, subkey, batch_size, n_channels, block_size)
            token_types = tokenizer.compute_token_types(xb, n_channels)

            model = GPT2_v3(vocab_size, n_embed, block_size, num_heads, num_layers, drop_rate, n_channels)
            dummy_x = jnp.zeros(shape=(batch_size, block_size), dtype=jnp.uint16)
            dummy_token_types = jnp.zeros_like(dummy_x)
            dummy_channel_ids = jnp.zeros_like(dummy_x)
            variables = model.init(rng_key, dummy_x, dummy_token_types, dummy_channel_ids)

            out = model.apply(variables, dummy_x, dummy_token_types, dummy_channel_ids)
            print("Test model.apply():", out.shape)

            print("Cross-entropy loss:", np.log(vocab_size))
            
            opt_state = optimizer.init(variables)
            losses = []
      else:
            model_name = MODEL_DIR / model_name
            print("Model:", model_name)

            # Load model
            with open(model_name, 'rb') as f:
                  model_file = pickle.load(f)

            globals().update(model_file)
            variables = model_file['variables']
            model = model_file['model']
            opt_state = model_file['opt_state']
            losses = model_file['losses']

      # Training loop
      train_step = make_train_step(model.apply, optimizer, n_channels)

      pbar = tqdm(range(epochs))
      for epoch in pbar:
            rng_key, subkey = jax.random.split(rng_key)

            xb, yb = tokenizer.get_token_batch(train_tokens, subkey, batch_size, n_channels, block_size)
            token_types = tokenizer.compute_token_types(xb, n_channels)
            channel_ids = tokenizer.compute_channel_ids(xb, n_channels)

            variables, opt_state, loss, is_nan = train_step(
                  variables,
                  opt_state,
                  xb,
                  yb,
                  token_types,
                  channel_ids,
            )

            # Bring scalar to host once per step (fine)
            loss_f = float(loss)
            losses.append(loss_f)
            
            pbar.set_description(f"Epoch: {epoch}, Loss: {loss :.4f}")

            model_file = {
                  "epochs": epochs,
                  "epoch": epoch,
                  "model": model,
                  "vocab_size": vocab_size,
                  "block_size": block_size,
                  "variables": variables,
                  "losses": losses,
                  "opt_state": opt_state,
                  "learning_rate": learning_rate,
                  "n_channels": n_channels,
                  "n_bins": n_bins,
                  "edges": edges, 
                  "mids": mids,
                  "mu": mu,
                  "sigma": sigma,
                  "resample_interval": resample_interval,
                  "zero_bin": zero_bin,
                  "s": s,
                  "x_max": x_max,
            }

            # Save model every 10th epoch or if last
            is_every_10 = (epoch + 1) % 10 == 0
            is_last = (epoch == epochs - 1)
            if is_every_10 or is_last:
                  with open(model_name, 'wb') as f:
                        pickle.dump(model_file, f)

      return model_file

