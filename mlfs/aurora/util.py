import os
import datetime
import time
import requests
import pandas as pd
import json
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import openmeteo_requests
import requests_cache
from retry_requests import retry
import hopsworks
import hsfs
from pathlib import Path

import pandas as pd
import datetime
from pathlib import Path


def get_historical_weather_sweden(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float
):
  
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "cloud_cover_mean",
            "precipitation_sum",
            "sunshine_duration"
        ]
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    daily = response.Daily()

    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s"),
            end=pd.to_datetime(daily.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ),
        "cloud_cover_mean": daily.Variables(0).ValuesAsNumpy(),
        "precipitation_sum": daily.Variables(1).ValuesAsNumpy(),
        "sunshine_duration": daily.Variables(2).ValuesAsNumpy()
    }

    df_weather = pd.DataFrame(daily_data).dropna()
    df_weather[["cloud_cover_mean", "precipitation_sum"]] = (
        df_weather[["cloud_cover_mean", "precipitation_sum"]].astype("float32")
    )

    return df_weather


def check_file_path(file_path):
    my_file = Path(file_path)
    if my_file.is_file() == False:
        print(f"Error. File not found at the path: {file_path} ")
    else:
        print(f"File successfully found at the path: {file_path}")


def get_kp(
    csv_path: str | Path,
    day: datetime.date,
) -> pd.DataFrame:
    """
    Returns a DataFrame with geomagnetic indices (Kp/ap) for a given day.

    Parameters
    ----------
    csv_path : str or Path
        Path to CSV file containing historical Kp/ap data.
    day : datetime.date
        Date for which to retrieve geomagnetic data.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame containing geomagnetic features for the given day.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing or no data exists for the given date.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Kp index file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {
        "YYYY", "MM", "DD",
        "Kp1", "Kp2", "Kp3", "Kp4", "Kp5", "Kp6", "Kp7", "Kp8",
        "ap1", "ap2", "ap3", "ap4", "ap5", "ap6", "ap7", "ap8",
        "Ap"
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Construct date
    df["date"] = pd.to_datetime(
        dict(year=df.YYYY, month=df.MM, day=df.DD)
    )

    # Filter by requested day
    day_ts = pd.to_datetime(day)
    day_df = df[df["date"] == day_ts]

    if day_df.empty:
        raise ValueError(f"No geomagnetic data found for date: {day}")

    # Keep only relevant columns
    feature_cols = [
        "date",
        "Kp1", "Kp2", "Kp3", "Kp4", "Kp5", "Kp6", "Kp7", "Kp8",
        "ap1", "ap2", "ap3", "ap4", "ap5", "ap6", "ap7", "ap8",
        "Ap"
    ]

    day_df = day_df[feature_cols].copy()

    # Ensure numeric types
    numeric_cols = [c for c in day_df.columns if c != "date"]
    day_df[numeric_cols] = day_df[numeric_cols].astype("float32")

    return day_df


def get_latest_complete_kp_from_nowcast() -> pd.DataFrame:
    # Return 7 days so we can do the lag features calculation
    import pandas as pd
    import requests
    from io import StringIO
    import datetime

    url = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_nowcast.txt"
    response = requests.get(url)
    response.raise_for_status()

    column_names = [
        "YYYY", "MM", "DD", "days", "days_m", "BSR", "dB",
        "Kp1", "Kp2", "Kp3", "Kp4", "Kp5", "Kp6", "Kp7", "Kp8",
        "ap1", "ap2", "ap3", "ap4", "ap5", "ap6", "ap7", "ap8",
        "Ap", "SN", "F10.7obs", "F10.7adj", "D"
    ]

    df = pd.read_csv(
        StringIO(response.text),
        delim_whitespace=True,
        comment="#",
        header=None,
        names=column_names,
    )

    if df.empty:
        raise ValueError("Nowcast file is empty")

    # Construct date
    df["date"] = pd.to_datetime(
        dict(year=df.YYYY, month=df.MM, day=df.DD)
    )
    
    df_last_7 = df.sort_values("date").tail(7)

    #today = pd.Timestamp(datetime.date.today())

    # 1) only completed days
    #df = df[df["date"] < today]

    # 2) drop rows with missing (-1) geomagnetic values
    geomagnetic_cols = (
        ["Kp1","Kp2","Kp3","Kp4","Kp5","Kp6","Kp7","Kp8"] +
        ["ap1","ap2","ap3","ap4","ap5","ap6","ap7","ap8"] +
        ["Ap"]
    )

    for col in geomagnetic_cols:
        df_last_7 = df_last_7[df_last_7[col] >= 0]

    if df_last_7.empty:
        raise ValueError("No complete geomagnetic day available in nowcast file")

    df_last_7 = df_last_7.sort_values("date")
    
    #df_last_7["ap"] = df_last_7["Ap"].astype("float32")
    df_last_7.columns = [c.lower() for c in df_last_7.columns]

    out = df_last_7[
        [
            "date",
            "kp1","kp2","kp3","kp4","kp5","kp6","kp7","kp8",
            "ap1","ap2","ap3","ap4","ap5","ap6","ap7","ap8",
            "ap",
        ]
    ].copy()

    # Latest valid day
    #row = df_last_7.sort_values("date").iloc[-1]

    """ out = pd.DataFrame(
        {
            "date": [row.date],
            "kp1": [row.Kp1],
            "kp2": [row.Kp2],
            "kp3": [row.Kp3],
            "kp4": [row.Kp4],
            "kp5": [row.Kp5],
            "kp6": [row.Kp6],
            "kp7": [row.Kp7],
            "kp8": [row.Kp8],
            "ap1": [row.ap1],
            "ap2": [row.ap2],
            "ap3": [row.ap3],
            "ap4": [row.ap4],
            "ap5": [row.ap5],
            "ap6": [row.ap6],
            "ap7": [row.ap7],
            "ap8": [row.ap8],
            "ap": [row.Ap],
        }
    ) """

    for col in out.columns:
        if col != "date":
            out[col] = out[col].astype("float32")

    return out


def fetch_newest_solar_data(run_date):

    run_date = pd.to_datetime(run_date).date()
    
    ### 1. Obtain Newest data 
    plasma_url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
    mag_url    = "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"

    plasma = requests.get(plasma_url).json()
    mag = requests.get(mag_url).json()

    plasma_df = pd.DataFrame(plasma[1:], columns=plasma[0])
    mag_df = pd.DataFrame(mag[1:], columns=mag[0])

    plasma_df["time_tag"] = pd.to_datetime(plasma_df["time_tag"])
    mag_df["time_tag"] = pd.to_datetime(mag_df["time_tag"])

    df = plasma_df.merge(
        mag_df[["time_tag", "bz_gsm"]],
        on="time_tag",
        how="inner",
    )

    df = df.rename(columns={
        "time_tag": "date",
        "speed": "vsw",
        "density": "density",
        "bz_gsm": "bz",
    })
    
    df = df.sort_values("date").ffill()
    df = df.dropna()
    df.drop(columns=["temperature"], inplace=True)

    numeric_cols = ["vsw", "density", "bz"]
    df[numeric_cols] = df[numeric_cols].astype("float32")
    
    df["pressure"] = 1.6726e-6 * df["density"] * (df["vsw"] ** 2)
    
    return df

def solar_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    for lag in [1, 2, 3]:
        df[f'vsw_lag{lag}'] = df['vsw'].shift(lag)
        df[f'bz_lag{lag}'] = df['bz'].shift(lag)
        df[f'pressure_lag{lag}'] = df['pressure'].shift(lag)

    df['bz_3d_mean'] = df['bz'].rolling(3).mean()
    df['bz_7d_min'] = df['bz'].rolling(7).min()

    df['vsw_3d_mean'] = df['vsw'].rolling(3).mean()
    df['pressure_3d_max'] = df['pressure'].rolling(3).max()

    df['vbz'] = df['vsw'] * df['bz']
    df['vbz_neg'] = df['vsw'] * df['bz'].clip(upper=0)

    float_cols = df.select_dtypes("float").columns
    df[float_cols] = df[float_cols].astype("float32")

    before = len(df)

    df = df.dropna().reset_index(drop=True)

    after = len(df)

    print(f"🧹 Dropped {before - after} rows due to NaNs")
    print(f"📊 Remaining rows: {after}")
    
    return df


def geomagnetic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    
    order = df["date"].sort_values().index
    df = df.loc[order].reset_index(drop=True)

    # Lagged Kp features (daily mean + max are most informative)
    kp_cols = [f"kp{i}" for i in range(1, 9)]
    df["kp_mean"] = df[kp_cols].mean(axis=1)
    df["kp_max"] = df[kp_cols].max(axis=1)

    for lag in [1, 2, 3]:
        df[f"kp_mean_lag_{lag}"] = df["kp_mean"].shift(lag)
        df[f"kp_max_lag_{lag}"] = df["kp_max"].shift(lag)

    return df