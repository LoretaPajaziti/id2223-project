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

    today = pd.Timestamp(datetime.date.today())

    # 1) only completed days
    df = df[df["date"] < today]

    # 2) drop rows with missing (-1) geomagnetic values
    geomagnetic_cols = (
        ["Kp1","Kp2","Kp3","Kp4","Kp5","Kp6","Kp7","Kp8"] +
        ["ap1","ap2","ap3","ap4","ap5","ap6","ap7","ap8"] +
        ["Ap"]
    )

    for col in geomagnetic_cols:
        df = df[df[col] >= 0]

    if df.empty:
        raise ValueError("No complete geomagnetic day available in nowcast file")

    # Latest valid day
    row = df.sort_values("date").iloc[-1]

    out = pd.DataFrame(
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
    )

    for col in out.columns:
        if col != "date":
            out[col] = out[col].astype("float32")

    return out
