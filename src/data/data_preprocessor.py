from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nifty50_data.csv"

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "nifty50_cleaned.csv"
)


# ============================================================
# DATA PREPROCESSING
# ============================================================

def preprocess_data():

    print("=" * 70)
    print("STARTING NIFTY 50 DATA PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. CHECK RAW DATA FILE
    # --------------------------------------------------------

    print("\n[1/8] Checking raw data file...")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {RAW_DATA_PATH}"
        )

    print(f"Raw data file found: {RAW_DATA_PATH}")

    # --------------------------------------------------------
    # 2. LOAD RAW DATA
    # --------------------------------------------------------

    print("\n[2/8] Loading raw CSV data...")

    df = pd.read_csv(RAW_DATA_PATH)

    print("Raw data loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # --------------------------------------------------------
    # 3. NORMALIZE COLUMN NAMES
    # --------------------------------------------------------

    print("\n[3/8] Normalizing column names...")

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    print(f"Columns after normalization: {list(df.columns)}")

    # --------------------------------------------------------
    # 4. VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------------

    print("\n[4/8] Validating required columns...")

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("All required columns are present.")

    # --------------------------------------------------------
    # 5. CONVERT DATA TYPES
    # --------------------------------------------------------

    print("\n[5/8] Converting data types...")

    # Convert Date column
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Remove timezone information if present
    if hasattr(df["Date"].dt, "tz") and df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)

    # Convert numeric columns
    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    if "Adj Close" in df.columns:
        numeric_columns.append("Adj Close")

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    print("Data type conversion completed.")

    # --------------------------------------------------------
    # 6. DATA QUALITY CHECKS
    # --------------------------------------------------------

    print("\n[6/8] Performing data quality checks...")

    initial_rows = len(df)

    # Remove rows with invalid dates
    invalid_dates = df["Date"].isna().sum()

    if invalid_dates > 0:

        print(f"Removing {invalid_dates} rows with invalid dates.")

        df = df.dropna(subset=["Date"])

    # Remove duplicate dates
    duplicate_dates = df["Date"].duplicated().sum()

    if duplicate_dates > 0:

        print(
            f"Removing {duplicate_dates} duplicate date records."
        )

        df = df.drop_duplicates(
            subset=["Date"],
            keep="last"
        )

    # Check missing values
    print("\nMissing values before handling:")

    missing_values = df.isnull().sum()

    print(missing_values)

    # Remove rows with missing critical market data
    critical_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing_critical = df[critical_columns].isnull().any(axis=1).sum()

    if missing_critical > 0:

        print(
            f"\nRemoving {missing_critical} rows "
            "with missing critical values."
        )

        df = df.dropna(
            subset=critical_columns
        )

    # --------------------------------------------------------
    # 7. VALIDATE MARKET DATA
    # --------------------------------------------------------

    print("\n[7/8] Validating market price consistency...")

    # High should not be lower than Open, Low or Close
    invalid_high = (
        (df["High"] < df["Open"]) |
        (df["High"] < df["Low"]) |
        (df["High"] < df["Close"])
    )

    # Low should not be higher than Open, High or Close
    invalid_low = (
        (df["Low"] > df["Open"]) |
        (df["Low"] > df["High"]) |
        (df["Low"] > df["Close"])
    )

    invalid_price_rows = invalid_high | invalid_low

    invalid_price_count = invalid_price_rows.sum()

    if invalid_price_count > 0:

        print(
            f"WARNING: Found {invalid_price_count} "
            "rows with inconsistent OHLC values."
        )

        print("\nInvalid rows:")

        print(
            df.loc[
                invalid_price_rows,
                ["Date", "Open", "High", "Low", "Close"]
            ]
        )

        df = df.loc[
            ~invalid_price_rows
        ].copy()

    else:

        print("OHLC price consistency check passed.")

    # Check negative values
    negative_values = (
        (df[numeric_columns] < 0).any(axis=1)
    )

    negative_count = negative_values.sum()

    if negative_count > 0:

        print(
            f"WARNING: Found {negative_count} "
            "rows containing negative numeric values."
        )

        df = df.loc[
            ~negative_values
        ].copy()

    # --------------------------------------------------------
    # 8. SORT DATA CHRONOLOGICALLY
    # --------------------------------------------------------

    print("\nSorting data chronologically...")

    df = df.sort_values(
        by="Date"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)

    print(f"Original rows: {initial_rows}")
    print(f"Final rows: {len(df)}")
    print(f"Rows removed: {initial_rows - len(df)}")

    print(f"First date: {df['Date'].min()}")
    print(f"Last date: {df['Date'].max()}")

    print("\nFinal missing values:")

    print(df.isnull().sum())

    print("\nFinal dataset information:")

    print(df.info())

    print("\nFirst 5 rows:")

    print(df.head())

    print("\nLast 5 rows:")

    print(df.tail())

    # --------------------------------------------------------
    # SAVE PROCESSED DATA
    # --------------------------------------------------------

    print("\nSaving processed dataset...")

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print(
        f"\nProcessed data saved successfully:\n"
        f"{PROCESSED_DATA_PATH}"
    )

    print("\n" + "=" * 70)
    print("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return df


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("Executing data preprocessing script...")

    processed_df = preprocess_data()

    print("\nPreprocessing script finished.")