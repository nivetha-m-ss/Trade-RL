"""
TD3 Diagnostic Analysis

Investigates why the trained TD3 agent may be choosing
low or zero investment allocations.

Outputs:
- Action distribution
- Allocation statistics
- Trade statistics
- Reward statistics
- Diagnostic CSV
- Action/allocation visualization
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import TD3


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.environment.trading_environment import TradingEnvironment


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "validation.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "trained"
    / "td3_nifty50_final.zip"
)

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "reports"
)

PLOTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "plots"
)


# ============================================================
# TD3 DIAGNOSTIC
# ============================================================

class TD3Diagnostics:

    def __init__(self):

        print("\n" + "=" * 80)
        print("INITIALIZING TD3 DIAGNOSTICS")
        print("=" * 80)

        RESULTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        PLOTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.environment = None

        self.model = None

        self.results = None

        print("\nDiagnostic initialized successfully.")

    # ========================================================
    # CREATE ENVIRONMENT
    # ========================================================

    def create_environment(self):

        print("\n" + "=" * 80)
        print("CREATING VALIDATION ENVIRONMENT")
        print("=" * 80)

        self.environment = TradingEnvironment(
            data_path=VALIDATION_DATA_PATH,
        )

        print("\nEnvironment created.")

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(self):

        print("\n" + "=" * 80)
        print("LOADING TRAINED TD3 MODEL")
        print("=" * 80)

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Model not found:\n{MODEL_PATH}"
            )

        print(f"\nModel Path: {MODEL_PATH}")

        self.model = TD3.load(
            str(MODEL_PATH),
            device="auto",
        )

        print("\nTD3 model loaded successfully.")

    # ========================================================
    # RUN DIAGNOSTIC
    # ========================================================

    def run_diagnostic(self):

        print("\n" + "=" * 80)
        print("RUNNING TD3 ACTION DIAGNOSTIC")
        print("=" * 80)

        observation, info = (
            self.environment.reset(
                seed=42
            )
        )

        print("\nInitial observation loaded.")

        print("\nPredicting actions across validation dataset...")

        records = []

        step = 0

        while True:

            # ------------------------------------------------
            # PREDICT ACTION
            # ------------------------------------------------

            action, _ = self.model.predict(
                observation,
                deterministic=True,
            )

            action_value = float(
                np.asarray(action).flatten()[0]
            )

            # ------------------------------------------------
            # EXECUTE ACTION
            # ------------------------------------------------

            observation, reward, terminated, truncated, info = (
                self.environment.step(
                    action
                )
            )

            step += 1

            # ------------------------------------------------
            # STORE RESULTS
            # ------------------------------------------------

            records.append(
                {
                    "Date": info["date"],
                    "Predicted_Action": action_value,
                    "Actual_Allocation": info["position"],
                    "Reward": reward,
                    "Portfolio_Value": info["portfolio_value"],
                    "Cash": info["cash"],
                    "Shares": info["shares"],
                    "Trade_Type": info["trade_type"],
                    "Transaction_Cost": info["transaction_cost"],
                }
            )

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if step % 50 == 0:

                print(
                    f"Step {step}: "
                    f"Action={action_value:.6f}, "
                    f"Allocation={info['position']:.2%}, "
                    f"Reward={reward:.6f}"
                )

            # ------------------------------------------------
            # TERMINATION
            # ------------------------------------------------

            if terminated or truncated:

                print(
                    "\nDiagnostic simulation completed."
                )

                break

        self.results = pd.DataFrame(
            records
        )

        self.results["Date"] = pd.to_datetime(
            self.results["Date"]
        )

        print(
            f"\nTotal Steps: {len(self.results)}"
        )

        return self.results

    # ========================================================
    # ANALYZE ACTIONS
    # ========================================================

    def analyze_actions(self):

        print("\n" + "=" * 80)
        print("ACTION ANALYSIS")
        print("=" * 80)

        actions = (
            self.results["Predicted_Action"]
        )

        allocations = (
            self.results["Actual_Allocation"]
        )

        print("\nACTION STATISTICS")

        print("-" * 60)

        print(
            f"Minimum Action       : {actions.min():.6f}"
        )

        print(
            f"Maximum Action       : {actions.max():.6f}"
        )

        print(
            f"Mean Action          : {actions.mean():.6f}"
        )

        print(
            f"Median Action        : {actions.median():.6f}"
        )

        print(
            f"Standard Deviation   : {actions.std():.6f}"
        )

        print("-" * 60)

        print("\nACTUAL ALLOCATION STATISTICS")

        print("-" * 60)

        print(
            f"Minimum Allocation   : {allocations.min():.2%}"
        )

        print(
            f"Maximum Allocation   : {allocations.max():.2%}"
        )

        print(
            f"Mean Allocation      : {allocations.mean():.2%}"
        )

        print(
            f"Median Allocation    : {allocations.median():.2%}"
        )

        print("-" * 60)

        # ----------------------------------------------------
        # ACTION DISTRIBUTION
        # ----------------------------------------------------

        print("\nACTION DISTRIBUTION")

        print("-" * 60)

        print(
            f"Near 0% Allocation   : "
            f"{(actions <= 0.01).sum()}"
        )

        print(
            f"Low Allocation (0-25%): "
            f"{((actions > 0.01) & (actions <= 0.25)).sum()}"
        )

        print(
            f"Medium (25-75%)      : "
            f"{((actions > 0.25) & (actions <= 0.75)).sum()}"
        )

        print(
            f"High (75-99%)        : "
            f"{((actions > 0.75) & (actions < 0.99)).sum()}"
        )

        print(
            f"Near 100% Allocation : "
            f"{(actions >= 0.99).sum()}"
        )

        print("-" * 60)

    # ========================================================
    # TRADE ANALYSIS
    # ========================================================

    def analyze_trades(self):

        print("\n" + "=" * 80)
        print("TRADE ANALYSIS")
        print("=" * 80)

        trade_counts = (
            self.results["Trade_Type"]
            .value_counts()
        )

        print("\nTrade Counts:")

        print(
            trade_counts.to_string()
        )

        total_trades = (
            self.results["Trade_Type"]
            .isin(["BUY", "SELL"])
            .sum()
        )

        print(
            f"\nTotal Trade Actions: {total_trades}"
        )

        print(
            f"Total Transaction Costs: "
            f"₹{self.results['Transaction_Cost'].sum():,.2f}"
        )

    # ========================================================
    # REWARD ANALYSIS
    # ========================================================

    def analyze_rewards(self):

        print("\n" + "=" * 80)
        print("REWARD ANALYSIS")
        print("=" * 80)

        rewards = (
            self.results["Reward"]
        )

        print("\nReward Statistics:")

        print("-" * 60)

        print(
            f"Minimum Reward      : {rewards.min():.6f}"
        )

        print(
            f"Maximum Reward      : {rewards.max():.6f}"
        )

        print(
            f"Mean Reward         : {rewards.mean():.6f}"
        )

        print(
            f"Median Reward       : {rewards.median():.6f}"
        )

        print(
            f"Positive Rewards    : {(rewards > 0).sum()}"
        )

        print(
            f"Negative Rewards    : {(rewards < 0).sum()}"
        )

        print(
            f"Zero Rewards        : {(rewards == 0).sum()}"
        )

        print("-" * 60)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    def save_results(self):

        print("\n" + "=" * 80)
        print("SAVING DIAGNOSTIC RESULTS")
        print("=" * 80)

        output_path = (
            RESULTS_DIRECTORY
            / "td3_action_diagnostics.csv"
        )

        self.results.to_csv(
            output_path,
            index=False,
        )

        print(
            f"\nDiagnostic CSV saved:"
            f"\n{output_path}"
        )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    def plot_diagnostics(self):

        print("\n" + "=" * 80)
        print("GENERATING TD3 DIAGNOSTIC VISUALIZATION")
        print("=" * 80)

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(16, 18),
        )

        # ----------------------------------------------------
        # ACTION
        # ----------------------------------------------------

        print("\nPlotting predicted actions...")

        axes[0].plot(
            self.results["Date"],
            self.results["Predicted_Action"],
            label="Predicted TD3 Action",
        )

        axes[0].set_title(
            "TD3 Predicted Actions"
        )

        axes[0].set_ylabel(
            "Action (0-1)"
        )

        axes[0].set_ylim(
            0,
            1,
        )

        axes[0].grid(
            True,
            alpha=0.3,
        )

        axes[0].legend()

        # ----------------------------------------------------
        # ACTUAL ALLOCATION
        # ----------------------------------------------------

        print("\nPlotting actual allocation...")

        axes[1].plot(
            self.results["Date"],
            self.results["Actual_Allocation"],
            label="Actual Portfolio Allocation",
        )

        axes[1].set_title(
            "Actual Portfolio Allocation"
        )

        axes[1].set_ylabel(
            "Allocation"
        )

        axes[1].set_ylim(
            0,
            1,
        )

        axes[1].grid(
            True,
            alpha=0.3,
        )

        axes[1].legend()

        # ----------------------------------------------------
        # REWARD
        # ----------------------------------------------------

        print("\nPlotting rewards...")

        axes[2].plot(
            self.results["Date"],
            self.results["Reward"],
            label="Daily Reward",
        )

        axes[2].set_title(
            "TD3 Daily Reward"
        )

        axes[2].set_ylabel(
            "Reward"
        )

        axes[2].grid(
            True,
            alpha=0.3,
        )

        axes[2].legend()

        # ----------------------------------------------------
        # PORTFOLIO VALUE
        # ----------------------------------------------------

        print("\nPlotting portfolio value...")

        axes[3].plot(
            self.results["Date"],
            self.results["Portfolio_Value"],
            label="TD3 Portfolio Value",
        )

        axes[3].axhline(
            100_000,
            linestyle="--",
            label="Initial Capital",
        )

        axes[3].set_title(
            "TD3 Portfolio Value"
        )

        axes[3].set_ylabel(
            "Portfolio Value (₹)"
        )

        axes[3].set_xlabel(
            "Date"
        )

        axes[3].grid(
            True,
            alpha=0.3,
        )

        axes[3].legend()

        # ----------------------------------------------------
        # LAYOUT
        # ----------------------------------------------------

        plt.tight_layout()

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        output_path = (
            PLOTS_DIRECTORY
            / "td3_action_diagnostics.png"
        )

        print("\nSaving visualization...")

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"\nVisualization saved:"
            f"\n{output_path}"
        )

        plt.show()

        plt.close()

    # ========================================================
    # RUN
    # ========================================================

    def run(self):

        print("\n")

        print("=" * 100)

        print(
            "TRADE RL - TD3 DIAGNOSTIC ANALYSIS"
        )

        print("=" * 100)

        # ----------------------------------------------------
        # ENVIRONMENT
        # ----------------------------------------------------

        print(
            "\nPHASE 1: CREATE ENVIRONMENT"
        )

        self.create_environment()

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        print(
            "\nPHASE 2: LOAD MODEL"
        )

        self.load_model()

        # ----------------------------------------------------
        # DIAGNOSTIC
        # ----------------------------------------------------

        print(
            "\nPHASE 3: RUN DIAGNOSTIC"
        )

        self.run_diagnostic()

        # ----------------------------------------------------
        # ANALYSIS
        # ----------------------------------------------------

        print(
            "\nPHASE 4: ANALYZE ACTIONS"
        )

        self.analyze_actions()

        print(
            "\nPHASE 5: ANALYZE TRADES"
        )

        self.analyze_trades()

        print(
            "\nPHASE 6: ANALYZE REWARDS"
        )

        self.analyze_rewards()

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        print(
            "\nPHASE 7: SAVE RESULTS"
        )

        self.save_results()

        # ----------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------

        print(
            "\nPHASE 8: GENERATE VISUALIZATION"
        )

        self.plot_diagnostics()

        # ----------------------------------------------------
        # COMPLETION
        # ----------------------------------------------------

        print("\n")

        print("=" * 100)

        print(
            "TD3 DIAGNOSTIC COMPLETED"
        )

        print("=" * 100)


# ============================================================
# MAIN
# ============================================================

def main():

    diagnostic = TD3Diagnostics()

    diagnostic.run()


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    main()