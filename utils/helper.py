import pandas as pd


def crop_datetime(df, start_datetime="", end_datetime="", print_data_info=False):

    if print_data_info:
        print(df.head())
        print(df.info())
        print(df.describe())

    start_dt = pd.to_datetime(start_datetime, utc=True)
    end_dt  = pd.to_datetime(end_datetime, utc=True)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        data_filtered = df[(df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)]
    else:
        df.index = pd.to_datetime(df.index, utc=True)
        data_filtered = df[(df.index >= start_dt) & (df.index <= end_dt)]
    
    return data_filtered