import os
import pandas as pd
import numpy as np
from scipy.stats import linregress

def extract_session_number(filename):
    digits = "".join(filter(str.isdigit, filename))
    return int(digits) if digits else None


def load_case_study_files(base_path, case_studies, data_type, min_session=None):
    all_dataframes = []
    reference_columns = None

    for case_study in case_studies:
        folder_path = os.path.join(base_path, case_study, data_type)
        print(f"Processing folder: {folder_path}")

        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])

        for i, filename in enumerate(file_list):
            session_number = extract_session_number(filename)

            if min_session is not None and session_number is not None:
                if session_number < min_session:
                    print(f"Skipping file: {filename}")
                    continue

            file_path = os.path.join(folder_path, filename)

            if reference_columns is None and not all_dataframes:
                print(f"Reading first file with headers: {file_path}")
                df = pd.read_csv(file_path)
                reference_columns = df.columns
            else:
                print(f"Reading file without headers: {file_path}")
                df = pd.read_csv(file_path, skiprows=1, header=None)
                df.columns = reference_columns

            df.iloc[:, 0] = df.iloc[:, 0].astype(str) + f"_{session_number}"
            df["session_number"] = session_number
            all_dataframes.append(df)

    if not all_dataframes:
        return pd.DataFrame()

    return pd.concat(all_dataframes, ignore_index=True)

def split_mean_difference(x, split_ratio=0.5):
    x = x.values
    split_index = int(len(x) * split_ratio)
    if split_index == 0 or split_index == len(x):
        return np.nan
    return x[:split_index].mean() - x[split_index:].mean()


def calculate_pattern_stats(
    df,
    value_col,
    username_col="username",
    group_col="group_number",
    time_col=None,
    prefix=""
):
    data = df.copy()

    if time_col is not None:
        data["timestamp"] = pd.to_datetime(data[time_col], errors="coerce")

    data["original_username"] = data[username_col]
    data[["uid", "session"]] = data[username_col].str.split("_", expand=True)
    data["session"] = data["session"].astype(int)

    data["normalized_rank"] = (
        data.groupby(["session", group_col])[value_col]
        .rank(method="min", ascending=True)
        / data.groupby(["session", group_col])[value_col].transform("count")
    )

    patrn9_name = f"{prefix}patrn9"
    user_avg_rank = (
        data.groupby("original_username")["normalized_rank"]
        .mean()
        .reset_index()
        .rename(columns={"normalized_rank": patrn9_name})
    )

    agg_dict = {
        f"{prefix}patrn1": (value_col, lambda x: np.std(x, ddof=1)),
        f"{prefix}patrn2": (value_col, "mean"),
        f"{prefix}patrn3": (value_col, "median"),
        f"{prefix}patrn4": (value_col, lambda x: np.sum(x >= 5) / len(x) if len(x) > 0 else 0),
        f"{prefix}patrn5": (value_col, lambda x: linregress(range(len(x)), x).slope if len(x) > 1 else np.nan),
        f"{prefix}patrn7": (value_col, lambda x: split_mean_difference(x, 0.5)),
        f"{prefix}patrn8": (value_col, lambda x: split_mean_difference(x, 0.7)),
        "num_ratings": (value_col, "count")
    }

    if time_col is not None:
        agg_dict[f"{prefix}patrn6"] = (
            "timestamp",
            lambda x: x.diff().mean().total_seconds() if len(x) > 1 else np.nan
        )

    user_stats = data.groupby("original_username").agg(**agg_dict).reset_index()

    user_stats[f"{prefix}patrn10"] = (
        user_stats[f"{prefix}patrn7"] / user_stats["num_ratings"]
    ).apply(lambda x: 1 if x > 0 else 0)

    user_stats[f"{prefix}patrn11"] = (
        user_stats[f"{prefix}patrn8"] / user_stats["num_ratings"]
    ).apply(lambda x: 1 if x > 0 else 0)

    final_stats = user_stats.merge(user_avg_rank, on="original_username")
    final_stats = final_stats.rename(columns={"original_username": username_col})
    final_stats = final_stats.drop(columns=["num_ratings"])

    return final_stats