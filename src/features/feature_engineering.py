from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty50_cleaned.csv"
)

OUTPUT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty50_features.csv"
)

PLOTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "plots"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features():

    print("=" * 70)
    print("STARTING FEATURE ENGINEERING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. CHECK INPUT FILE
    # --------------------------------------------------------

    print("\n[1/10] Checking processed dataset...")

    if not INPUT_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Processed dataset not found:\n{INPUT_DATA_PATH}"
        )

    print(f"Input file found: {INPUT_DATA_PATH}")

    # --------------------------------------------------------
    # 2. LOAD DATA
    # --------------------------------------------------------

    print("\n[2/10] Loading processed dataset...")

    df = pd.read_csv(INPUT_DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # --------------------------------------------------------
    # 3. VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------------

    print("\n[3/10] Validating required columns...")

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close"
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

    print("Required columns are available.")

    # --------------------------------------------------------
    # 4. SORT DATA CHRONOLOGICALLY
    # --------------------------------------------------------

    print("\n[4/10] Sorting data chronologically...")

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    print("Data sorted successfully.")

    # --------------------------------------------------------
    # 5. CALCULATE RETURNS
    # --------------------------------------------------------

    print("\n[5/10] Calculating price returns...")

    # Simple daily return
    df["Daily_Return"] = (
        df["Close"].pct_change()
    )

    # Logarithmic daily return
    df["Log_Return"] = np.log(
        df["Close"] / df["Close"].shift(1)
    )

    print("Daily return calculated.")
    print("Log return calculated.")

    # --------------------------------------------------------
    # 6. CALCULATE MOVING AVERAGES
    # --------------------------------------------------------

    print("\n[6/10] Calculating moving averages...")

    # Simple Moving Averages
    df["SMA_5"] = (
        df["Close"].rolling(window=5).mean()
    )

    df["SMA_20"] = (
        df["Close"].rolling(window=20).mean()
    )

    df["SMA_50"] = (
        df["Close"].rolling(window=50).mean()
    )

    # Exponential Moving Averages
    df["EMA_12"] = (
        df["Close"].ewm(
            span=12,
            adjust=False
        ).mean()
    )

    df["EMA_26"] = (
        df["Close"].ewm(
            span=26,
            adjust=False
        ).mean()
    )

    # Price relative to moving averages
    df["Close_SMA5_Ratio"] = (
        df["Close"] / df["SMA_5"]
    )

    df["Close_SMA20_Ratio"] = (
        df["Close"] / df["SMA_20"]
    )

    df["Close_SMA50_Ratio"] = (
        df["Close"] / df["SMA_50"]
    )

    df["Close_EMA12_Ratio"] = (
        df["Close"] / df["EMA_12"]
    )

    df["Close_EMA26_Ratio"] = (
        df["Close"] / df["EMA_26"]
    )

    print("SMA and EMA calculations completed.")

    # --------------------------------------------------------
    # 7. CALCULATE RSI
    # --------------------------------------------------------

    print("\n[7/10] Calculating RSI...")

    rsi_period = 14

    price_change = df["Close"].diff()

    gains = price_change.clip(lower=0)

    losses = -price_change.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / rsi_period,
        min_periods=rsi_period,
        adjust=False
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / rsi_period,
        min_periods=rsi_period,
        adjust=False
    ).mean()

    rs = average_gain / average_loss

    df["RSI_14"] = (
        100 - (100 / (1 + rs))
    )

    print("RSI calculation completed.")

    # --------------------------------------------------------
    # 8. CALCULATE MACD
    # --------------------------------------------------------

    print("\n[8/10] Calculating MACD...")

    df["MACD"] = (
        df["EMA_12"] - df["EMA_26"]
    )

    df["MACD_Signal"] = (
        df["MACD"].ewm(
            span=9,
            adjust=False
        ).mean()
    )

    df["MACD_Histogram"] = (
        df["MACD"] - df["MACD_Signal"]
    )

    # Normalize MACD relative to the current price
    df["MACD_Pct"] = (
        df["MACD"] / df["Close"]
    )

    df["MACD_Signal_Pct"] = (
        df["MACD_Signal"] / df["Close"]
    )

    df["MACD_Histogram_Pct"] = (
        df["MACD_Histogram"] / df["Close"]
    )

    print("MACD calculations completed.")

    # --------------------------------------------------------
    # 9. CALCULATE VOLATILITY FEATURES
    # --------------------------------------------------------

    print("\n[9/10] Calculating volatility indicators...")

    # Bollinger Bands
    bollinger_period = 20

    bollinger_std_multiplier = 2

    df["BB_Middle"] = (
        df["Close"].rolling(
            window=bollinger_period
        ).mean()
    )

    rolling_std = (
        df["Close"].rolling(
            window=bollinger_period
        ).std()
    )

    df["BB_Upper"] = (
        df["BB_Middle"]
        + bollinger_std_multiplier * rolling_std
    )

    df["BB_Lower"] = (
        df["BB_Middle"]
        - bollinger_std_multiplier * rolling_std
    )

    # Bollinger Band Width
    df["BB_Width"] = (
        (df["BB_Upper"] - df["BB_Lower"])
        / df["BB_Middle"]
    )

    # Bollinger %B
    df["BB_Percent_B"] = (
        (df["Close"] - df["BB_Lower"])
        / (df["BB_Upper"] - df["BB_Lower"])
    )

    # Rolling volatility
    df["Volatility_5"] = (
        df["Daily_Return"]
        .rolling(window=5)
        .std()
    )

    df["Volatility_20"] = (
        df["Daily_Return"]
        .rolling(window=20)
        .std()
    )

    df["Volatility_50"] = (
        df["Daily_Return"]
        .rolling(window=50)
        .std()
    )

    # Average True Range
    previous_close = df["Close"].shift(1)

    true_range_1 = (
        df["High"] - df["Low"]
    )

    true_range_2 = (
        abs(df["High"] - previous_close)
    )

    true_range_3 = (
        abs(df["Low"] - previous_close)
    )

    df["True_Range"] = pd.concat(
        [
            true_range_1,
            true_range_2,
            true_range_3
        ],
        axis=1
    ).max(axis=1)

    df["ATR_14"] = (
        df["True_Range"]
        .rolling(window=14)
        .mean()
    )

    # Normalize ATR relative to price
    df["ATR_14_Pct"] = (
        df["ATR_14"] / df["Close"]
    )

    print("Bollinger Bands calculated.")
    print("Rolling volatility calculated.")
    print("ATR calculated.")

    # --------------------------------------------------------
    # FEATURE LIST
    # --------------------------------------------------------

    feature_columns = [
        "Daily_Return",
        "Log_Return",
        "SMA_5",
        "SMA_20",
        "SMA_50",
        "EMA_12",
        "EMA_26",
        "Close_SMA5_Ratio",
        "Close_SMA20_Ratio",
        "Close_SMA50_Ratio",
        "Close_EMA12_Ratio",
        "Close_EMA26_Ratio",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "MACD_Pct",
        "MACD_Signal_Pct",
        "MACD_Histogram_Pct",
        "BB_Middle",
        "BB_Upper",
        "BB_Lower",
        "BB_Width",
        "BB_Percent_B",
        "Volatility_5",
        "Volatility_20",
        "Volatility_50",
        "True_Range",
        "ATR_14",
        "ATR_14_Pct"
    ]

    # --------------------------------------------------------
    # 10. DATA VALIDATION
    # --------------------------------------------------------

    print("\n[10/10] Validating engineered features...")

    print("\nFeature columns generated:")

    for feature in feature_columns:

        print(f"  - {feature}")

    print(f"\nTotal engineered features: {len(feature_columns)}")

    # Missing values generated by rolling calculations
    print("\nChecking missing values...")

    missing_values = df[feature_columns].isnull().sum()

    print(missing_values)

    rows_before = len(df)

    df = df.dropna(
        subset=feature_columns
    ).reset_index(drop=True)

    rows_removed = rows_before - len(df)

    print(f"\nRows before removing NaN values: {rows_before}")
    print(f"Rows removed: {rows_removed}")
    print(f"Rows remaining: {len(df)}")

    # Check infinite values
    print("\nChecking infinite values...")

    numeric_features = df[feature_columns]

    infinite_values = np.isinf(
        numeric_features.to_numpy()
    ).sum()

    print(f"Infinite values found: {infinite_values}")

    if infinite_values > 0:

        print("Replacing infinite values with NaN...")

        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        df = df.dropna(
            subset=feature_columns
        ).reset_index(drop=True)

    # --------------------------------------------------------
    # SAVE FEATURE DATASET
    # --------------------------------------------------------

    print("\nSaving feature-engineered dataset...")

    OUTPUT_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_DATA_PATH,
        index=False
    )

    print(
        f"\nFeature dataset saved successfully:\n"
        f"{OUTPUT_DATA_PATH}"
    )

    # --------------------------------------------------------
    # GENERATE VISUALIZATIONS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATING DATA VISUALIZATIONS")
    print("=" * 70)

    generate_visualizations(df)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)

    print(f"Original rows: {rows_before}")
    print(f"Final rows: {len(df)}")
    print(f"Rows removed: {rows_removed}")

    print(f"\nFirst date: {df['Date'].min()}")
    print(f"Last date: {df['Date'].max()}")

    print(f"\nTotal columns: {len(df.columns)}")
    print(f"Total engineered features: {len(feature_columns)}")

    print("\nFinal dataset columns:")

    print(df.columns.tolist())

    print("\nFirst 5 rows:")

    print(df.head())

    print("\nLast 5 rows:")

    print(df.tail())

    print("\nDataset information:")

    df.info()

    print(
        f"\nFeature-engineered dataset:\n"
        f"{OUTPUT_DATA_PATH}"
    )

    print(
        f"\nVisualizations saved in:\n"
        f"{PLOTS_DIRECTORY}"
    )

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return df


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def generate_visualizations(df):

    print("\nPreparing visualization directory...")

    PLOTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. PRICE + MOVING AVERAGES
    # --------------------------------------------------------

    print("\n[Visualization 1/6] Price and Moving Averages...")

    plt.figure(figsize=(16, 8))

    plt.plot(
        df["Date"],
        df["Close"],
        label="Nifty 50 Close",
        linewidth=1.5
    )

    plt.plot(
        df["Date"],
        df["SMA_20"],
        label="SMA 20",
        linewidth=1
    )

    plt.plot(
        df["Date"],
        df["SMA_50"],
        label="SMA 50",
        linewidth=1
    )

    plt.title(
        "Nifty 50 Price with Moving Averages"
    )

    plt.xlabel("Date")
    plt.ylabel("Index Value")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIRECTORY / "price_moving_averages.png",
        dpi=300
    )

    plt.close()

    print("Saved: price_moving_averages.png")

    # --------------------------------------------------------
    # 2. RSI
    # --------------------------------------------------------

    print("\n[Visualization 2/6] RSI...")

    plt.figure(figsize=(16, 6))

    plt.plot(
        df["Date"],
        df["RSI_14"],
        label="RSI 14",
        linewidth=1.2
    )

    plt.axhline(
        70,
        linestyle="--",
        label="Overbought (70)"
    )

    plt.axhline(
        30,
        linestyle="--",
        label="Oversold (30)"
    )

    plt.axhline(
        50,
        linestyle=":",
        label="Neutral (50)"
    )

    plt.title("Nifty 50 RSI - Relative Strength Index")

    plt.xlabel("Date")
    plt.ylabel("RSI")

    plt.ylim(0, 100)

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIRECTORY / "rsi_14.png",
        dpi=300
    )

    plt.close()

    print("Saved: rsi_14.png")

    # --------------------------------------------------------
    # 3. MACD
    # --------------------------------------------------------

    print("\n[Visualization 3/6] MACD...")

    fig, ax = plt.subplots(
        figsize=(16, 7)
    )

    ax.plot(
        df["Date"],
        df["MACD"],
        label="MACD",
        linewidth=1.2
    )

    ax.plot(
        df["Date"],
        df["MACD_Signal"],
        label="Signal",
        linewidth=1.2
    )

    ax.bar(
        df["Date"],
        df["MACD_Histogram"],
        label="Histogram",
        alpha=0.5
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_title("Nifty 50 MACD")

    ax.set_xlabel("Date")
    ax.set_ylabel("MACD")

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIRECTORY / "macd.png",
        dpi=300
    )

    plt.close()

    print("Saved: macd.png")

    # --------------------------------------------------------
    # 4. BOLLINGER BANDS
    # --------------------------------------------------------

    print("\n[Visualization 4/6] Bollinger Bands...")

    plt.figure(figsize=(16, 8))

    plt.plot(
        df["Date"],
        df["Close"],
        label="Close",
        linewidth=1.2
    )

    plt.plot(
        df["Date"],
        df["BB_Middle"],
        label="Middle Band",
        linewidth=1
    )

    plt.plot(
        df["Date"],
        df["BB_Upper"],
        label="Upper Band",
        linewidth=1
    )

    plt.plot(
        df["Date"],
        df["BB_Lower"],
        label="Lower Band",
        linewidth=1
    )

    plt.fill_between(
        df["Date"],
        df["BB_Lower"],
        df["BB_Upper"],
        alpha=0.15
    )

    plt.title("Nifty 50 Bollinger Bands")

    plt.xlabel("Date")
    plt.ylabel("Index Value")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIRECTORY / "bollinger_bands.png",
        dpi=300
    )

    plt.close()

    print("Saved: bollinger_bands.png")

    # --------------------------------------------------------
    # 5. DAILY RETURNS DISTRIBUTION
    # --------------------------------------------------------

    print("\n[Visualization 5/6] Daily Returns Distribution...")

    plt.figure(figsize=(12, 7))

    plt.hist(
        df["Daily_Return"].dropna(),
        bins=50,
        edgecolor="black",
        alpha=0.7
    )

    plt.axvline(
        df["Daily_Return"].mean(),
        linestyle="--",
        label="Mean Return"
    )

    plt.title("Nifty 50 Daily Returns Distribution")

    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIRECTORY / "daily_returns_distribution.png",
        dpi=300
    )

    plt.close()

    print("Saved: daily_returns_distribution.png")

    # --------------------------------------------------------
    # 6. ROLLING VOLATILITY
    # --------------------------------------------------------

    print("\n[Visualization 6/6] Rolling Volatility...")

    plt.figure(figsize=(16, 7))

    plt.plot(
        df["Date"],
        df["Volatility_5"],
        label="5-Day Volatility",
        linewidth=1
    )

    plt.plot(
        df["Date"],
        df["Volatility_20"],
        label="20-Day Volatility",
        linewidth=1
    )

    plt.plot(
        df["Date"],
        df["Volatility_50"],
        label="50-Day Volatility",
        linewidth=1
    )

    plt.title("Nifty 50 Rolling Volatility")

    plt.xlabel("Date")
    plt.ylabel("Volatility")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIRECTORY / "rolling_volatility.png",
        dpi=300
    )

    plt.close()

    print("Saved: rolling_volatility.png")

    # --------------------------------------------------------
    # VISUALIZATION COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL VISUALIZATIONS GENERATED")
    print("=" * 70)

    print(f"\nPlots directory:\n{PLOTS_DIRECTORY}")

    print("\nGenerated files:")

    plot_files = [
        "price_moving_averages.png",
        "rsi_14.png",
        "macd.png",
        "bollinger_bands.png",
        "daily_returns_distribution.png",
        "rolling_volatility.png"
    ]

    for plot_file in plot_files:

        print(f"  - {plot_file}")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("Executing feature engineering script...")

    feature_df = engineer_features()

    print("\nFeature engineering script finished.")