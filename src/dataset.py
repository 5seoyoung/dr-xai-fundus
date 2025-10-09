# src/dataset.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def load_aptos_df(data_root):
    data_root = Path(data_root)
    csv = data_root / "train.csv"
    df = pd.read_csv(csv)  # id_code, diagnosis(0~4)
    df["image_file"] = df["id_code"].astype(str) + ".png"
    df["binary"] = (df["diagnosis"] >= 2).astype(int)
    return df, data_root / "train_images"

def split_df(df, seed=42, val_size=0.1, test_size=0.1):
    df_tr, df_te = train_test_split(df, test_size=test_size, stratify=df["binary"], random_state=seed)
    df_tr, df_va = train_test_split(df_tr, test_size=val_size/(1-test_size), stratify=df_tr["binary"], random_state=seed)
    return df_tr.reset_index(drop=True), df_va.reset_index(drop=True), df_te.reset_index(drop=True)
