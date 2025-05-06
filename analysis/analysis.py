
# -*- coding: utf-8 -*-
"""
Helpers for the paper
"""
import os
import pandas as pd
from scipy import stats

def read_data(folder: "stats"):
    data = []
    for model in ["ultra", "simkgc", "ilp"]:
        df = pd.read_csv(os.path.join(folder, f"best_metric_per_eta_{model}.csv"), index_col=0)
        if model == "ultra":
            df["text"] = 0
        else:
            df["text"] = 1
        data.append(df)
    return pd.concat(data)

def print_corr(df, cols_param, cols_metric):
    for col in cols_param:
        for m in cols_metric:
            res = stats.spearmanr(df[col], df[m])
            print(f"{col.upper()}:\t vs. {m.upper()}: {res.statistic:.4f}, p={res.pvalue:.4f}")


if __name__ == "__main__":
    DF = read_data("stats")
    print_corr(df=DF, cols_param=["text"], cols_metric=["MRR", "H@1", "H@3", "H@10"])
