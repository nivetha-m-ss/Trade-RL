from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "discretized"

PLOT_DIR = PROJECT_ROOT / "results" / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES = [
    "Daily_Return",
    "RSI_14",
    "MACD_Histogram_Pct",
    "Close_SMA20_Ratio",
    "Volatility_20",
]

NUM_BINS = 5

# Fixed bins for RSI
RSI_BINS = [
    0,
    20,
    40,
    60,
    80,
    100,
]

# Fixed bins for Price / SMA20
PRICE_SMA20_BINS = [
    -np.inf,
    0.98,
    0.995,
    1.005,
    1.02,
    np.inf,
]


# ============================================================
# STATE DISCRETIZER
# ============================================================

class StateDiscretizer:

    def __init__(self):

        self.bin_edges = {}

        print("\n" + "=" * 70)
        print("STATE DISCRETIZER INITIALIZED")
        print("=" * 70)

        print(f"Features: {FEATURES}")
        print(f"Number of bins: {NUM_BINS}")

    # ========================================================
    # LOAD DATA
    # ========================================================

    def load_data(self, filename):

        filepath = SPLIT_DIR / filename

        print(f"\nLoading dataset:")
        print(filepath)

        if not filepath.exists():

            raise FileNotFoundError(
                f"Dataset not found: {filepath}"
            )

        df = pd.read_csv(filepath)

        print(f"Loaded rows: {len(df):,}")
        print(f"Columns: {df.columns.tolist()}")

        return df

    # ========================================================
    # VALIDATE FEATURES
    # ========================================================

    def validate_features(self, df):

        print("\nValidating required features...")

        missing_features = [
            feature
            for feature in FEATURES
            if feature not in df.columns
        ]

        if missing_features:

            raise ValueError(
                f"Missing required features: {missing_features}"
            )

        print("All required features are available.")

    # ========================================================
    # FIT BIN BOUNDARIES USING TRAIN DATA
    # ========================================================

    def fit(self, train_df):

        print("\n" + "=" * 70)
        print("FITTING DISCRETIZATION BOUNDARIES")
        print("=" * 70)

        self.validate_features(train_df)

        print("\nCalculating boundaries using TRAINING DATA only...")

        # ----------------------------------------------------
        # Daily Return
        # ----------------------------------------------------

        print("\nProcessing Daily_Return...")

        daily_return_edges = np.quantile(
            train_df["Daily_Return"].dropna(),
            np.linspace(0, 1, NUM_BINS + 1)
        )

        daily_return_edges = np.unique(daily_return_edges)

        daily_return_edges[0] = -np.inf
        daily_return_edges[-1] = np.inf

        self.bin_edges["Daily_Return"] = daily_return_edges

        print(f"Bins: {daily_return_edges}")

        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        print("\nProcessing RSI_14...")

        self.bin_edges["RSI_14"] = np.array(RSI_BINS)

        print(f"Bins: {RSI_BINS}")

        # ----------------------------------------------------
        # MACD Histogram Percentage
        # ----------------------------------------------------

        print("\nProcessing MACD_Histogram_Pct...")

        macd_edges = np.quantile(
            train_df["MACD_Histogram_Pct"].dropna(),
            np.linspace(0, 1, NUM_BINS + 1)
        )

        macd_edges = np.unique(macd_edges)

        macd_edges[0] = -np.inf
        macd_edges[-1] = np.inf

        self.bin_edges["MACD_Histogram_Pct"] = macd_edges

        print(f"Bins: {macd_edges}")

        # ----------------------------------------------------
        # Close / SMA20 Ratio
        # ----------------------------------------------------

        print("\nProcessing Close_SMA20_Ratio...")

        self.bin_edges["Close_SMA20_Ratio"] = np.array(
            PRICE_SMA20_BINS
        )

        print(f"Bins: {PRICE_SMA20_BINS}")

        # ----------------------------------------------------
        # Volatility
        # ----------------------------------------------------

        print("\nProcessing Volatility_20...")

        volatility_edges = np.quantile(
            train_df["Volatility_20"].dropna(),
            np.linspace(0, 1, NUM_BINS + 1)
        )

        volatility_edges = np.unique(volatility_edges)

        volatility_edges[0] = -np.inf
        volatility_edges[-1] = np.inf

        self.bin_edges["Volatility_20"] = volatility_edges

        print(f"Bins: {volatility_edges}")

        print("\nDiscretization boundaries fitted successfully.")

        return self

    # ========================================================
    # TRANSFORM DATA
    # ========================================================

    def transform(self, df, dataset_name="dataset"):

        print("\n" + "=" * 70)
        print(f"DISCRETIZING {dataset_name.upper()} DATA")
        print("=" * 70)

        self.validate_features(df)

        result = df.copy()

        # ----------------------------------------------------
        # Daily Return
        # ----------------------------------------------------

        print("\nDiscretizing Daily_Return...")

        result["Daily_Return_State"] = pd.cut(
            result["Daily_Return"],
            bins=self.bin_edges["Daily_Return"],
            labels=False,
            include_lowest=True
        )

        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        print("Discretizing RSI_14...")

        result["RSI_State"] = pd.cut(
            result["RSI_14"],
            bins=self.bin_edges["RSI_14"],
            labels=False,
            include_lowest=True
        )

        # ----------------------------------------------------
        # MACD
        # ----------------------------------------------------

        print("Discretizing MACD_Histogram_Pct...")

        result["MACD_State"] = pd.cut(
            result["MACD_Histogram_Pct"],
            bins=self.bin_edges["MACD_Histogram_Pct"],
            labels=False,
            include_lowest=True
        )

        # ----------------------------------------------------
        # Price / SMA20
        # ----------------------------------------------------

        print("Discretizing Close_SMA20_Ratio...")

        result["Trend_State"] = pd.cut(
            result["Close_SMA20_Ratio"],
            bins=self.bin_edges["Close_SMA20_Ratio"],
            labels=False,
            include_lowest=True
        )

        # ----------------------------------------------------
        # Volatility
        # ----------------------------------------------------

        print("Discretizing Volatility_20...")

        result["Volatility_State"] = pd.cut(
            result["Volatility_20"],
            bins=self.bin_edges["Volatility_20"],
            labels=False,
            include_lowest=True
        )

        # ----------------------------------------------------
        # MARKET STATE
        # ----------------------------------------------------

        state_columns = [
            "Daily_Return_State",
            "RSI_State",
            "MACD_State",
            "Trend_State",
            "Volatility_State",
        ]

        print("\nCreating Market_State...")

        result["Market_State"] = result[state_columns].apply(
            lambda row: tuple(row.astype(int)),
            axis=1
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        print("\nValidating discretized states...")

        for column in state_columns:

            invalid_values = result[column].isna().sum()

            if invalid_values > 0:

                raise ValueError(
                    f"{column} contains {invalid_values} invalid values"
                )

        print("All states successfully generated.")

        return result

    # ========================================================
    # SAVE BIN CONFIGURATION
    # ========================================================

    def save_bins(self):

        print("\nSaving discretization boundaries...")

        output_file = OUTPUT_DIR / "state_bins.json"

        serializable_bins = {}

        for feature, edges in self.bin_edges.items():

            serializable_bins[feature] = [
                str(edge) if np.isinf(edge) else float(edge)
                for edge in edges
            ]

        with open(output_file, "w") as file:

            json.dump(
                serializable_bins,
                file,
                indent=4
            )

        print(f"Saved bin configuration:")
        print(output_file)

    # ========================================================
    # SAVE DATA
    # ========================================================

    def save_data(self, df, dataset_name):

        output_file = OUTPUT_DIR / f"{dataset_name}_states.csv"

        print(f"\nSaving {dataset_name} states...")

        df.to_csv(
            output_file,
            index=False
        )

        print(f"Saved successfully:")
        print(output_file)

    # ========================================================
    # VISUALIZATION
    # ========================================================

    def create_visualizations(self, train_df, discretized_train):

        print("\n" + "=" * 70)
        print("GENERATING STATE DISCRETIZATION VISUALIZATIONS")
        print("=" * 70)

        # ----------------------------------------------------
        # Feature distributions with bin boundaries
        # ----------------------------------------------------

        print("\nGenerating feature bin visualization...")

        fig, axes = plt.subplots(
            5,
            1,
            figsize=(14, 18)
        )

        features = FEATURES

        for i, feature in enumerate(features):

            ax = axes[i]

            train_df[feature].hist(
                bins=50,
                ax=ax
            )

            ax.set_title(
                f"{feature} Distribution"
            )

            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")

            # Show bin boundaries
            edges = self.bin_edges[feature]

            for edge in edges:

                if np.isfinite(edge):

                    ax.axvline(
                        edge,
                        linestyle="--",
                        alpha=0.7
                    )

        plt.tight_layout()

        output_file = PLOT_DIR / "state_discretization_bins.png"

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved: {output_file}")

        # ----------------------------------------------------
        # State frequency distribution
        # ----------------------------------------------------

        print("\nGenerating state frequency visualization...")

        state_columns = [
            "Daily_Return_State",
            "RSI_State",
            "MACD_State",
            "Trend_State",
            "Volatility_State",
        ]

        fig, axes = plt.subplots(
            5,
            1,
            figsize=(14, 18)
        )

        for i, column in enumerate(state_columns):

            state_counts = discretized_train[column].value_counts().sort_index()

            axes[i].bar(
                state_counts.index.astype(str),
                state_counts.values
            )

            axes[i].set_title(
                f"{column} Frequency"
            )

            axes[i].set_xlabel("State")
            axes[i].set_ylabel("Frequency")

        plt.tight_layout()

        output_file = PLOT_DIR / "state_frequency_distribution.png"

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved: {output_file}")

        # ----------------------------------------------------
        # Combined market state distribution
        # ----------------------------------------------------

        print("\nGenerating combined market state distribution...")

        state_counts = discretized_train["Market_State"].value_counts()

        fig, ax = plt.subplots(
            figsize=(14, 8)
        )

        ax.bar(
            range(len(state_counts)),
            state_counts.values
        )

        ax.set_title(
            "Market State Frequency Distribution"
        )

        ax.set_xlabel("Market State Index")
        ax.set_ylabel("Frequency")

        ax.set_xticks([])

        plt.tight_layout()

        output_file = PLOT_DIR / "market_state_distribution.png"

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved: {output_file}")

    # ========================================================
    # SUMMARY
    # ========================================================

    def print_summary(self, train_states, validation_states, test_states):

        print("\n" + "=" * 70)
        print("STATE DISCRETIZATION SUMMARY")
        print("=" * 70)

        print("\nDataset sizes:")

        print(f"Train:      {len(train_states):,}")
        print(f"Validation:  {len(validation_states):,}")
        print(f"Test:       {len(test_states):,}")

        print("\nFeature states:")

        for feature in FEATURES:

            print(f"\n{feature}")

            print(f"Boundaries: {self.bin_edges[feature]}")

        print("\nUnique market states:")

        print(
            f"Train: {train_states['Market_State'].nunique():,}"
        )

        print(
            f"Validation: {validation_states['Market_State'].nunique():,}"
        )

        print(
            f"Test: {test_states['Market_State'].nunique():,}"
        )

        print("\nState examples:")

        print(
            train_states[
                [
                    "Date",
                    "Daily_Return_State",
                    "RSI_State",
                    "MACD_State",
                    "Trend_State",
                    "Volatility_State",
                    "Market_State"
                ]
            ].head(10)
        )


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("TRADE RL - STATE DISCRETIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Initialize
    # --------------------------------------------------------

    discretizer = StateDiscretizer()

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nSTEP 1: Loading datasets")

    train_df = discretizer.load_data("train.csv")

    validation_df = discretizer.load_data("validation.csv")

    test_df = discretizer.load_data("test.csv")

    # --------------------------------------------------------
    # Fit only on train
    # --------------------------------------------------------

    print("\nSTEP 2: Fit discretization using training data")

    discretizer.fit(train_df)

    # --------------------------------------------------------
    # Transform all datasets
    # --------------------------------------------------------

    print("\nSTEP 3: Discretize training dataset")

    train_states = discretizer.transform(
        train_df,
        "train"
    )

    print("\nSTEP 4: Discretize validation dataset")

    validation_states = discretizer.transform(
        validation_df,
        "validation"
    )

    print("\nSTEP 5: Discretize test dataset")

    test_states = discretizer.transform(
        test_df,
        "test"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\nSTEP 6: Save discretized datasets")

    discretizer.save_data(
        train_states,
        "train"
    )

    discretizer.save_data(
        validation_states,
        "validation"
    )

    discretizer.save_data(
        test_states,
        "test"
    )

    discretizer.save_bins()

    # --------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------

    print("\nSTEP 7: Generate visualizations")

    discretizer.create_visualizations(
        train_df,
        train_states
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nSTEP 8: Print final summary")

    discretizer.print_summary(
        train_states,
        validation_states,
        test_states
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STATE DISCRETIZATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated files:")

    print("\nData:")
    print(OUTPUT_DIR / "train_states.csv")
    print(OUTPUT_DIR / "validation_states.csv")
    print(OUTPUT_DIR / "test_states.csv")
    print(OUTPUT_DIR / "state_bins.json")

    print("\nVisualizations:")
    print(PLOT_DIR / "state_discretization_bins.png")
    print(PLOT_DIR / "state_frequency_distribution.png")
    print(PLOT_DIR / "market_state_distribution.png")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()