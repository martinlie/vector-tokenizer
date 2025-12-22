# GPT-2

We will build a GPT-2 language model for predicting the next token in a sequence of time series vectors. 


### References:
- [1] [LLM from scratch using jax](https://github.com/ChristianOrr/transformers)


## Installation

```
python -m venv .venv
.venv/Scripts/activate
python -m pip install -r requirements.txt
```

## Run

First run load_data.py



## Loading data for the examples

We use an open Kaggle dataset for the examples in this project. Specifically, we utilize the [Delhi 5-Minute Electricity Demand for Forecasting dataset](https://www.kaggle.com/datasets/yug201/delhi-5-minute-electricity-demand-for-forecasting), which provides high-frequency electricity demand data for Delhi. This dataset is ideal for demonstrating time series analysis, forecasting techniques, and anomaly detection. By applying various modeling approaches, including PCA and machine learning, we explore patterns in electricity consumption and showcase the capabilities of TimeScape in extracting insights from temporal data.

The dataset consists of the following columns:

* datetime: Timestamp of the observation
* Power demand: Electricity demand (in kW) recorded every 5 minutes.
* temp: Temperature (°C).
* dwpt: Dew point temperature (°C).
* rhum: Relative humidity (%).
* wdir: Wind direction (degrees).
* wspd: Wind speed (m/s).
* pres: Atmospheric pressure (hPa).
* year, month, day, hour, minute: Breakdown of the timestamp for easy time-series analysis.

## Parameter Selection

The Parameters used below are a scaled down version of GPT-2. GPT-2 has 4 different sizes, small, medium, large and xl. This GPT-2 could be considered an extra-small version. Note that these models may not be able to fit into RAM on your device. The exact specifications of the different sized models are shown below:

### GPT-2 Small
- n_embed: 768
- block_size: 1024
- num_heads: 12
- num_layers: 12
- vocab_size: 50257 (uses Tiktoken vocab)

### GPT-2 Medium
- n_embed: 1024
- block_size: 1024
- num_heads: 16
- num_layers: 24
- vocab_size: 50257 (uses Tiktoken vocab)

### GPT-2 Large
- n_embed: 1280
- block_size: 1024
- num_heads: 20
- num_layers: 36
- vocab_size: 50257 (uses Tiktoken vocab)

### GPT-2 XL
- n_embed: 1600
- block_size: 1024
- num_heads: 25
- num_layers: 48
- vocab_size: 50257 (uses Tiktoken vocab)