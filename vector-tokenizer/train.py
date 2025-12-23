import pandas as pd
import numpy as np
import jax
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
learning_rate = 1e-5
epochs = 100000

rng_key = jax.random.PRNGKey(42)

if __name__ == "__main__":
      
      # Load splitted data
      X, Y = tl.get_split_data()
      print(f"X: {X.shape}")
      print(f"Y: {Y.shape}")

      # Z-norm standardisation
      mu = X.mean(axis=0)
      sigma = X.std(axis=0) + 1e-8
      X_normalized = (X-mu)/sigma

      # Derivation
      X_normalized = X_normalized.diff().dropna()
      X_nv = X_normalized.values.astype(np.float32)   # (N, D)

      # Discretization
      #edges, mids = discretize.fit_quantile_bins_global(X_nv, n_bins=n_bins)
      edges, mids = discretize.fit_equal_width_bins_global(X_nv, n_bins=n_bins)

      # encode
      #X_tok = discretize.encode_quantile_global(jnp.asarray(X_nv), edges)
      X_tok = discretize.encode_equal_width_global(jnp.asarray(X_nv), edges)

      # decode
      #X_rec = discretize.decode_quantile_global(X_tok, mids)
      X_rec = discretize.decode_equal_width_global(X_tok, mids)

      ZERO_BIN = int(np.argmin(np.abs(mids)))
      #or
      assert(ZERO_BIN == discretize.encode_equal_width_global(jnp.asarray([0]), edges))

      print("Zero midpoint:", mids[ZERO_BIN])
      print("Index:", ZERO_BIN)

      # Tokenize
      n_channels = len(X.columns)
      #tokens = tokenizer.encode_with_channels(X_tok, n_channels)
      tokens = tokenizer.encode_with_channels_sparse(X_tok, n_channels, ZERO_BIN)

      DATA_OFFSET = 2 + n_channels  # BOS+EOS+n_channels = 9
      vocab_size = DATA_OFFSET + n_bins

      T = tokens.shape[0]
      print("Total tokens:", T)
      print("Vocab size:", vocab_size)

      # Inspect first few tokens
      print("Last 100 tokens:", tokens[:100])

      # Continue training
      model_name = "token_model_20251222_140751.pkl"
      # else model_name = None

      # Train
      model_file = tl.train(model_name, rng_key, epochs, learning_rate, 
            tokens, mu, sigma, 
            batch_size, n_channels, block_size, n_embed, num_heads, num_layers, drop_rate, 
            vocab_size, n_bins, edges, mids)

