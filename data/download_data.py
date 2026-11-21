from pathlib import Path
from datetime import datetime, timedelta
import sys

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
import os


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

PROJECT_NAME = os.getenv("PROJECT_NAME", "Trade_RL")
SYMBOL = os.getenv("SYMBOL", "^NSEI")
DATA_SOURCE = os.getenv("DATA_SOURCE", "yfinance")

START_DATE = os.getenv("START_DATE", "2021-01-01")
END_DATE = os.getenv("END_DATE", "").strip()

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "nifty50_data.csv"


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_market_data():
    print("=" * 60)
    print(f"PROJECT       : {PROJECT_NAME}")
    print(f"DATA SOURCE   : {DATA_SOURCE}")
    print(f"SYMBOL        : {SYMBOL}")
    print(f"START DATE    : {START_DATE}")
    print(f"END DATE      : {END_DATE if END_DATE else 'Latest available'}")
    print("=" * 60)

    try:
        # Convert dates
        start_date = datetime.strptime(START_DATE, "%Y-%m-%d")

        # Yahoo Finance treats end_date as exclusive.
        # Add one day so the configured end date is included.
        if END_DATE:
            end_date = datetime.strptime(END_DATE, "%Y-%m-%d")
            end_date = end_date + timedelta(days=1)
            end_date = end_date.strftime("%Y-%m-%d")
        else:
            end_date = None

        print("\n[INFO] Starting market data download...")

        data = yf.download(
            tickers=SYMBOL,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date,
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False
        )

        print("[INFO] Download request completed.")

        # ----------------------------------------------------
        # Validate downloaded data
        # ----------------------------------------------------

        if data is None or data.empty:
            raise ValueError(
                "No market data received from Yahoo Finance."
            )

        print(f"[INFO] Downloaded rows: {len(data)}")

        # ----------------------------------------------------
        # Handle MultiIndex columns
        # ----------------------------------------------------

        if isinstance(data.columns, pd.MultiIndex):
            print("[INFO] Flattening MultiIndex columns...")

            data.columns = [
                column[0] if isinstance(column, tuple) else column
                for column in data.columns
            ]

        # ----------------------------------------------------
        # Normalize column names
        # ----------------------------------------------------

        data.columns = [
            str(column).strip()
            for column in data.columns
        ]

        # ----------------------------------------------------
        # Handle Date index
        # ----------------------------------------------------

        data.index = pd.to_datetime(data.index)

        # Remove timezone information if present
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        data.index.name = "Date"

        # ----------------------------------------------------
        # Validate required market columns
        # ----------------------------------------------------

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        # ----------------------------------------------------
        # Remove completely empty rows
        # ----------------------------------------------------

        data = data.dropna(
            subset=required_columns,
            how="all"
        )

        # ----------------------------------------------------
        # Remove duplicate dates
        # ----------------------------------------------------

        data = data[~data.index.duplicated(keep="last")]

        # ----------------------------------------------------
        # Sort chronologically
        # ----------------------------------------------------

        data = data.sort_index()

        # ----------------------------------------------------
        # Validate final dataset
        # ----------------------------------------------------

        if data.empty:
            raise ValueError(
                "Market dataset is empty after preprocessing."
            )

        # ----------------------------------------------------
        # Create output directory
        # ----------------------------------------------------

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Save raw market data
        # ----------------------------------------------------

        data.to_csv(OUTPUT_FILE)

        # ----------------------------------------------------
        # Display download summary
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("[SUCCESS] MARKET DATA DOWNLOADED")
        print("=" * 60)

        print(f"Symbol          : {SYMBOL}")
        print(f"Total Records   : {len(data)}")
        print(f"First Date      : {data.index.min().date()}")
        print(f"Last Date       : {data.index.max().date()}")
        print(f"Columns         : {list(data.columns)}")
        print(f"Output File     : {OUTPUT_FILE}")

        print("\n[INFO] First 5 records:")
        print(data.head())

        print("\n[INFO] Last 5 records:")
        print(data.tail())

        print("\n[INFO] Dataset information:")
        print(data.info())

        print("\n[INFO] Missing values:")
        print(data.isnull().sum())

        print("\n" + "=" * 60)

        return data

    except Exception as error:
        print("\n" + "=" * 60)
        print("[ERROR] DATA DOWNLOAD FAILED")
        print("=" * 60)

        print(f"Error Type : {type(error).__name__}")
        print(f"Error      : {error}")

        print("\nPlease verify:")
        print("1. Internet connection")
        print("2. Yahoo Finance availability")
        print("3. Symbol configuration in .env")
        print("4. START_DATE configuration")
        print("5. Required Python dependencies")

        print("=" * 60)

        sys.exit(1)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("[INFO] Trade RL Data Collection Started")

    download_market_data()

    print("\n[INFO] Data collection process completed.")