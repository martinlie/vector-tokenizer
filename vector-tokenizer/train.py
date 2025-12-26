import pandas as pd
import numpy as np
import jax, jaxlib
import jax.numpy as jnp
import tokenizer_lib as tl
import discretize_func as discretize
import tokenizer_func as tokenizer

# Parameters
n_bins = 4096 # discretization fit bins
n_embed = 768 # 16 # 32 # Number of embedding dimensions
batch_size = 8 #4 # How many independent sequences will we process in parallel?
block_size = 480 # What is the maximum context length for predictions?
num_heads = 12 #4 # Number of heads in the multi-headed block
num_layers = 12 #6 # Number of transformer decoder blocks
drop_rate = 0.1 # Dropout rate for regularization
learning_rate = 3e-5 #1e-5
epochs = 10000
resample_interval = "h"  # or "15min"

rng_key = jax.random.PRNGKey(42)

if __name__ == "__main__":
      tl.status()

      # Load splitted data
      X, Y = tl.get_split_data(resample_interval=resample_interval)
      print(f"X: {X.shape}")
      print(f"Y: {Y.shape}")

      # Z-norm standardisation
      mu = X.mean(axis=0)
      sigma = X.std(axis=0) + 1e-8
      X_normalized = (X-mu)/sigma

      # Derivation
      X_normalized = X_normalized.diff().dropna()
      X_nv = X_normalized.values.astype(np.float32)   # (N, D)

      # Discretization binning parameters
      # In z-space, a good default is s=1.0
      s = 1.0

      # pick coverage in z-space (e.g. cover +/- 6 stdev deltas)
      x_max = 6.0
      u_max = float(np.arcsinh(x_max / s))

      # IMPORTANT: n_bins must be odd to guarantee exact zero bin
      if n_bins % 2 == 0:
            n_bins += 1

      # build bins in u-space
      edges, mids, zero_bin = discretize.make_asinh_bins_with_zero(n_bins=n_bins, u_max=u_max)
      #edges, mids = discretize.fit_quantile_bins_global(X_nv, n_bins=n_bins)
      #edges, mids = discretize.fit_equal_width_bins_global(X_nv, n_bins=n_bins)

      # encode
      X_tok = discretize.encode_asinh_global(jnp.asarray(X_nv), edges, s=s)
      #X_tok = discretize.encode_quantile_global(jnp.asarray(X_nv), edges)
      #X_tok = discretize.encode_equal_width_global(jnp.asarray(X_nv), edges)

      # decode
      X_rec = discretize.decode_asinh_global(X_tok, mids, s=s)
      #X_rec = discretize.decode_quantile_global(X_tok, mids)
      #X_rec = discretize.decode_equal_width_global(X_tok, mids)

      # sanity: exact zero bin
      assert float(mids[zero_bin]) == 0.0
      #ZERO_BIN = int(np.argmin(np.abs(mids)))
      #or
      #assert(ZERO_BIN == discretize.encode_equal_width_global(jnp.asarray([0]), edges))

      print("Zero midpoint:", mids[zero_bin])
      print("Index:", zero_bin)

      # Tokenize
      n_channels = len(X.columns)
      #tokens = tokenizer.encode_with_channels(X_tok, n_channels)
      tokens = tokenizer.encode_with_channels_sparse(X_tok, n_channels, zero_bin)

      DATA_OFFSET = 2 + n_channels  # BOS+EOS+n_channels = 9
      vocab_size = DATA_OFFSET + n_bins

      T = tokens.shape[0]
      print("Total tokens:", T)
      print("Vocab size:", vocab_size)

      # Inspect first few tokens
      print("Last 100 tokens:", tokens[:100])

      # Continue training
      #model_name = "token_model_20251222_140751.pkl" # 15 min data interval
      #model_name = "token_model_20251223_095047.pkl" # hourly data interval
      model_name = None

      # Train
      model_file = tl.train(model_name, rng_key, epochs, learning_rate, 
            tokens, mu, sigma, resample_interval,
            batch_size, n_channels, block_size, n_embed, num_heads, num_layers, drop_rate, 
            vocab_size, n_bins, edges, mids, zero_bin, s, x_max)

