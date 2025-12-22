import os
from pathlib import Path
import pandas as pd
import kagglehub

# Download latest version
path = kagglehub.dataset_download("yug201/delhi-5-minute-electricity-demand-for-forecasting")

print("Path to dataset files:", path)

data = pd.read_csv(path + "/powerdemand_5min_2021_to_2024_with weather.csv")

data.datetime = pd.to_datetime(data['datetime'])
data = data.set_index("datetime", drop=True)
data = data.drop(columns=["Unnamed: 0", "minute"])

DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

data.to_parquet(DATA_DIR / "dehli.parquet")