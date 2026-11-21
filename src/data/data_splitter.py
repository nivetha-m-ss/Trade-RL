from pathlib import Path

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
    / "nifty50_features.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

PLOTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "plots"
)


# ============================================================
# SPLIT CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.70

VALIDATION_RATIO = 0.15

TEST_RATIO = 0.15


# ============================================================
# DATA SPLITTING
# ============================================================

def split_dataset():

    print("=" * 70)
    print("STARTING TRAIN / VALIDATION / TEST SPLITTING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. CHECK INPUT FILE
    # --------------------------------------------------------

    print("\n[1/7] Checking feature dataset...")

    if not INPUT_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Feature dataset not found:\n{INPUT_DATA_PATH}"
        )

    print(f"Input dataset found: {INPUT_DATA_PATH}")

    # --------------------------------------------------------
    # 2. LOAD DATA
    # --------------------------------------------------------

    print("\n[2/7] Loading feature dataset...")

    df = pd.read_csv(INPUT_DATA_PATH)

    print("Dataset loaded successfully.")

    print(f"Total rows: {len(df)}")

    print(f"Total columns: {len(df.columns)}")

    # --------------------------------------------------------
    # 3. VALIDATE DATA
    # --------------------------------------------------------

    print("\n[3/7] Validating dataset...")

    if "Date" not in df.columns:

        raise ValueError(
            "Date column is missing from feature dataset."
        )

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    if df["Date"].duplicated().any():

        raise ValueError(
            "Duplicate dates detected in feature dataset."
        )

    if TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO != 1.0:

        raise ValueError(
            "Train, validation, and test ratios must sum to 1."
        )

    print("Dataset validation completed.")

    # --------------------------------------------------------
    # 4. CALCULATE SPLIT INDICES
    # --------------------------------------------------------

    print("\n[4/7] Calculating chronological split boundaries...")

    total_rows = len(df)

    train_end_index = int(
        total_rows * TRAIN_RATIO
    )

    validation_end_index = int(
        total_rows * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    print(f"Total rows: {total_rows}")

    print(f"Training boundary: {train_end_index}")

    print(f"Validation boundary: {validation_end_index}")

    # --------------------------------------------------------
    # 5. SPLIT DATA
    # --------------------------------------------------------

    print("\n[5/7] Splitting dataset chronologically...")

    train_df = df.iloc[
        :train_end_index
    ].copy()

    validation_df = df.iloc[
        train_end_index:validation_end_index
    ].copy()

    test_df = df.iloc[
        validation_end_index:
    ].copy()

    print("Chronological splitting completed.")

    # --------------------------------------------------------
    # 6. VALIDATE SPLITS
    # --------------------------------------------------------

    print("\n[6/7] Validating generated datasets...")

    if len(train_df) == 0:

        raise ValueError(
            "Training dataset is empty."
        )

    if len(validation_df) == 0:

        raise ValueError(
            "Validation dataset is empty."
        )

    if len(test_df) == 0:

        raise ValueError(
            "Test dataset is empty."
        )

    # Verify chronology
    if train_df["Date"].max() >= validation_df["Date"].min():

        raise ValueError(
            "Training and validation dates overlap."
        )

    if validation_df["Date"].max() >= test_df["Date"].min():

        raise ValueError(
            "Validation and test dates overlap."
        )

    print("All split validations passed.")

    # --------------------------------------------------------
    # 7. SAVE DATASETS
    # --------------------------------------------------------

    print("\n[7/7] Saving datasets...")

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    train_path = (
        OUTPUT_DIRECTORY
        / "train.csv"
    )

    validation_path = (
        OUTPUT_DIRECTORY
        / "validation.csv"
    )

    test_path = (
        OUTPUT_DIRECTORY
        / "test.csv"
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    validation_df.to_csv(
        validation_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    print(f"\nTraining dataset saved:\n{train_path}")

    print(f"\nValidation dataset saved:\n{validation_path}")

    print(f"\nTest dataset saved:\n{test_path}")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA SPLITTING SUMMARY")
    print("=" * 70)

    print(
        f"\n{'Dataset':<15}"
        f"{'Rows':<15}"
        f"{'Percentage':<15}"
        f"{'Start Date':<20}"
        f"{'End Date'}"
    )

    print("-" * 80)

    print(
        f"{'Training':<15}"
        f"{len(train_df):<15}"
        f"{len(train_df) / total_rows * 100:.2f}%{'':<8}"
        f"{train_df['Date'].min().date()}{'':<10}"
        f"{train_df['Date'].max().date()}"
    )

    print(
        f"{'Validation':<15}"
        f"{len(validation_df):<15}"
        f"{len(validation_df) / total_rows * 100:.2f}%{'':<8}"
        f"{validation_df['Date'].min().date()}{'':<10}"
        f"{validation_df['Date'].max().date()}"
    )

    print(
        f"{'Test':<15}"
        f"{len(test_df):<15}"
        f"{len(test_df) / total_rows * 100:.2f}%{'':<8}"
        f"{test_df['Date'].min().date()}{'':<10}"
        f"{test_df['Date'].max().date()}"
    )

    print("-" * 80)

    print(f"\nTotal rows: {total_rows}")

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    print("\nGenerating data split visualization...")

    generate_split_visualization(
        train_df,
        validation_df,
        test_df
    )

    print("\n" + "=" * 70)
    print("DATA SPLITTING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return train_df, validation_df, test_df


# ============================================================
# VISUALIZATION
# ============================================================

def generate_split_visualization(
    train_df,
    validation_df,
    test_df
):

    PLOTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(16, 8))

    # Training data
    plt.plot(
        train_df["Date"],
        train_df["Close"],
        label="Training Data",
        linewidth=1.5
    )

    # Validation data
    plt.plot(
        validation_df["Date"],
        validation_df["Close"],
        label="Validation Data",
        linewidth=1.5
    )

    # Test data
    plt.plot(
        test_df["Date"],
        test_df["Close"],
        label="Test Data",
        linewidth=1.5
    )

    # Split boundaries
    plt.axvline(
        validation_df["Date"].min(),
        linestyle="--",
        label="Train / Validation Boundary"
    )

    plt.axvline(
        test_df["Date"].min(),
        linestyle="--",
        label="Validation / Test Boundary"
    )

    plt.title(
        "Nifty 50 Chronological Train / Validation / Test Split"
    )

    plt.xlabel("Date")

    plt.ylabel("Nifty 50 Index Value")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        PLOTS_DIRECTORY
        / "train_validation_test_split.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Split visualization saved:\n{output_path}"
    )


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("Executing dataset splitting script...")

    train_df, validation_df, test_df = split_dataset()

    print("\nDataset splitting script finished.")