"""
TD3 Model Evaluation for Trade RL

Responsibilities:
1. Load trained TD3 model
2. Evaluate model on validation dataset
3. Evaluate Buy & Hold benchmark
4. Calculate performance metrics
5. Compare strategies
6. Generate visualizations
7. Save evaluation results
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import TD3


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

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

INITIAL_CAPITAL = 100_000

TRANSACTION_COST = 0.001

TRADING_DAYS_PER_YEAR = 252

SEED = 42


# ============================================================
# TD3 EVALUATOR
# ============================================================

class TD3Evaluator:

    def __init__(
        self,
        validation_data_path=VALIDATION_DATA_PATH,
        model_path=MODEL_PATH,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
    ):

        print("\n" + "=" * 80)
        print("INITIALIZING TD3 EVALUATOR")
        print("=" * 80)

        self.validation_data_path = Path(
            validation_data_path
        )

        self.model_path = Path(
            model_path
        )

        self.initial_capital = float(
            initial_capital
        )

        self.transaction_cost = float(
            transaction_cost
        )

        self.td3_environment = None

        self.buy_hold_environment = None

        self.model = None

        self.td3_history = None

        self.buy_hold_history = None

        self.metrics = None

        # ----------------------------------------------------
        # CREATE DIRECTORIES
        # ----------------------------------------------------

        print("\nCreating result directories...")

        RESULTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        PLOTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("Directories created successfully.")

        print("\nTD3 Evaluator initialized.")

        print("=" * 80)

    # ========================================================
    # CREATE ENVIRONMENTS
    # ========================================================

    def create_environments(self):

        print("\n" + "=" * 80)
        print("CREATING VALIDATION ENVIRONMENTS")
        print("=" * 80)

        # ----------------------------------------------------
        # TD3 ENVIRONMENT
        # ----------------------------------------------------

        print("\nCreating TD3 validation environment...")

        self.td3_environment = TradingEnvironment(
            data_path=self.validation_data_path,
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost,
        )

        # ----------------------------------------------------
        # BUY AND HOLD ENVIRONMENT
        # ----------------------------------------------------

        print("\nCreating Buy & Hold environment...")

        self.buy_hold_environment = TradingEnvironment(
            data_path=self.validation_data_path,
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost,
        )

        print("\nValidation environments created.")

        print("=" * 80)

    # ========================================================
    # LOAD TRAINED MODEL
    # ========================================================

    def load_model(self):

        print("\n" + "=" * 80)
        print("LOADING TRAINED TD3 MODEL")
        print("=" * 80)

        if self.td3_environment is None:

            raise RuntimeError(
                "Create environments before loading the model."
            )

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Trained model not found:\n"
                f"{self.model_path}"
            )

        print(f"\nModel Path: {self.model_path}")

        print("\nLoading TD3 model...")

        self.model = TD3.load(
            str(self.model_path),
            env=self.td3_environment,
            device="auto",
        )

        print("\nModel loaded successfully.")

        print("=" * 80)

    # ========================================================
    # EVALUATE TD3
    # ========================================================

    def evaluate_td3(self):

        print("\n" + "=" * 80)
        print("EVALUATING TD3 ON VALIDATION DATA")
        print("=" * 80)

        if self.model is None:

            raise RuntimeError(
                "Load the trained model before evaluation."
            )

        print("\nResetting TD3 environment...")

        observation, info = (
            self.td3_environment.reset(
                seed=SEED
            )
        )

        print(
            f"Initial Portfolio Value: "
            f"₹{info['portfolio_value']:,.2f}"
        )

        print("\nStarting deterministic evaluation...")

        step = 0

        while True:

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            action, _ = self.model.predict(
                observation,
                deterministic=True,
            )

            # ------------------------------------------------
            # ENVIRONMENT STEP
            # ------------------------------------------------

            observation, reward, terminated, truncated, info = (
                self.td3_environment.step(
                    action
                )
            )

            step += 1

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if step % 50 == 0:

                print(
                    f"\nEvaluation Step: {step}"
                )

                print(
                    f"Date: {info['date']}"
                )

                print(
                    f"Action: {info['action']:.4f}"
                )

                print(
                    f"Allocation: "
                    f"{info['position']:.2%}"
                )

                print(
                    f"Portfolio Value: "
                    f"₹{info['portfolio_value']:,.2f}"
                )

            # ------------------------------------------------
            # TERMINATION
            # ------------------------------------------------

            if terminated or truncated:

                print("\nTD3 evaluation completed.")

                break

        print(
            f"\nTotal Evaluation Steps: {step}"
        )

        self.td3_history = pd.DataFrame(
            self.td3_environment.portfolio_history
        )

        print(
            f"TD3 History Records: "
            f"{len(self.td3_history):,}"
        )

        return self.td3_history

    # ========================================================
    # EVALUATE BUY AND HOLD
    # ========================================================

    def evaluate_buy_and_hold(self):

        print("\n" + "=" * 80)
        print("EVALUATING BUY & HOLD BENCHMARK")
        print("=" * 80)

        print("\nResetting Buy & Hold environment...")

        observation, info = (
            self.buy_hold_environment.reset(
                seed=SEED
            )
        )

        print(
            f"Initial Portfolio Value: "
            f"₹{info['portfolio_value']:,.2f}"
        )

        print("\nExecuting Buy & Hold strategy...")

        step = 0

        # ----------------------------------------------------
        # ALWAYS TARGET 100% INVESTMENT
        # ----------------------------------------------------

        action = np.array(
            [1.0],
            dtype=np.float32,
        )

        while True:

            observation, reward, terminated, truncated, info = (
                self.buy_hold_environment.step(
                    action
                )
            )

            step += 1

            if step % 100 == 0:

                print(
                    f"Benchmark Step: {step}"
                )

                print(
                    f"Portfolio Value: "
                    f"₹{info['portfolio_value']:,.2f}"
                )

            if terminated or truncated:

                print("\nBuy & Hold evaluation completed.")

                break

        self.buy_hold_history = pd.DataFrame(
            self.buy_hold_environment.portfolio_history
        )

        print(
            f"\nBuy & Hold History Records: "
            f"{len(self.buy_hold_history):,}"
        )

        return self.buy_hold_history

    # ========================================================
    # PREPARE HISTORY
    # ========================================================

    def prepare_history(
        self,
        history,
        strategy_name,
    ):

        print(
            f"\nPreparing {strategy_name} history..."
        )

        history = history.copy()

        history["Date"] = pd.to_datetime(
            history["Date"]
        )

        history = history.sort_values(
            "Date"
        )

        history = history.reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # PORTFOLIO RETURNS
        # ----------------------------------------------------

        history["Daily_Return"] = (
            history["Portfolio_Value"]
            .pct_change()
        )

        history["Daily_Return"] = (
            history["Daily_Return"]
            .fillna(0.0)
        )

        # ----------------------------------------------------
        # RUNNING PEAK
        # ----------------------------------------------------

        history["Running_Peak"] = (
            history["Portfolio_Value"]
            .cummax()
        )

        # ----------------------------------------------------
        # DRAWDOWN
        # ----------------------------------------------------

        history["Drawdown"] = (
            history["Portfolio_Value"]
            / history["Running_Peak"]
        ) - 1.0

        print(
            f"{strategy_name} history prepared."
        )

        return history

    # ========================================================
    # PERFORMANCE METRICS
    # ========================================================

    def calculate_metrics(
        self,
        history,
        strategy_name,
    ):

        print(
            f"\nCalculating metrics for {strategy_name}..."
        )

        portfolio_values = (
            history["Portfolio_Value"]
        )

        daily_returns = (
            history["Daily_Return"]
        )

        # ----------------------------------------------------
        # TOTAL RETURN
        # ----------------------------------------------------

        final_value = float(
            portfolio_values.iloc[-1]
        )

        total_return = (
            final_value
            / self.initial_capital
        ) - 1.0

        # ----------------------------------------------------
        # CAGR
        # ----------------------------------------------------

        number_of_periods = max(
            len(history) - 1,
            1,
        )

        years = (
            number_of_periods
            / TRADING_DAYS_PER_YEAR
        )

        if years > 0 and final_value > 0:

            cagr = (
                final_value
                / self.initial_capital
            ) ** (
                1 / years
            ) - 1.0

        else:

            cagr = 0.0

        # ----------------------------------------------------
        # MAXIMUM DRAWDOWN
        # ----------------------------------------------------

        running_peak = (
            portfolio_values.cummax()
        )

        drawdowns = (
            portfolio_values
            / running_peak
        ) - 1.0

        maximum_drawdown = float(
            drawdowns.min()
        )

        # ----------------------------------------------------
        # SHARPE RATIO
        # ----------------------------------------------------

        daily_return_std = (
            daily_returns.std()
        )

        if (
            daily_return_std is not None
            and daily_return_std > 0
        ):

            sharpe_ratio = (
                daily_returns.mean()
                / daily_return_std
            ) * np.sqrt(
                TRADING_DAYS_PER_YEAR
            )

        else:

            sharpe_ratio = 0.0

        # ----------------------------------------------------
        # VOLATILITY
        # ----------------------------------------------------

        annualized_volatility = (
            daily_return_std
            * np.sqrt(
                TRADING_DAYS_PER_YEAR
            )
        )

        # ----------------------------------------------------
        # TRANSACTION COST
        # ----------------------------------------------------

        if strategy_name == "TD3":

            total_transaction_cost = (
                self.td3_environment.total_transaction_cost
            )

            total_trades = (
                self.td3_environment.total_trades
            )

        else:

            total_transaction_cost = (
                self.buy_hold_environment.total_transaction_cost
            )

            total_trades = (
                self.buy_hold_environment.total_trades
            )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        metrics = {

            "Strategy": strategy_name,

            "Initial_Capital": (
                self.initial_capital
            ),

            "Final_Portfolio_Value": (
                final_value
            ),

            "Total_Return": (
                total_return
            ),

            "CAGR": (
                cagr
            ),

            "Maximum_Drawdown": (
                maximum_drawdown
            ),

            "Sharpe_Ratio": (
                sharpe_ratio
            ),

            "Annualized_Volatility": (
                annualized_volatility
            ),

            "Total_Trades": (
                total_trades
            ),

            "Total_Transaction_Cost": (
                total_transaction_cost
            ),
        }

        print(
            f"\n{strategy_name} Metrics Calculated."
        )

        return metrics

    # ========================================================
    # COMPARE STRATEGIES
    # ========================================================

    def compare_strategies(self):

        print("\n" + "=" * 80)
        print("COMPARING TD3 AGAINST BUY & HOLD")
        print("=" * 80)

        # ----------------------------------------------------
        # PREPARE HISTORY
        # ----------------------------------------------------

        self.td3_history = (
            self.prepare_history(
                self.td3_history,
                "TD3",
            )
        )

        self.buy_hold_history = (
            self.prepare_history(
                self.buy_hold_history,
                "Buy & Hold",
            )
        )

        # ----------------------------------------------------
        # CALCULATE METRICS
        # ----------------------------------------------------

        td3_metrics = (
            self.calculate_metrics(
                self.td3_history,
                "TD3",
            )
        )

        buy_hold_metrics = (
            self.calculate_metrics(
                self.buy_hold_history,
                "Buy & Hold",
            )
        )

        # ----------------------------------------------------
        # COMPARISON TABLE
        # ----------------------------------------------------

        self.metrics = pd.DataFrame(
            [
                td3_metrics,
                buy_hold_metrics,
            ]
        )

        print("\n" + "=" * 100)

        print(
            "STRATEGY PERFORMANCE COMPARISON"
        )

        print("=" * 100)

        display_columns = [
            "Strategy",
            "Final_Portfolio_Value",
            "Total_Return",
            "CAGR",
            "Maximum_Drawdown",
            "Sharpe_Ratio",
            "Annualized_Volatility",
            "Total_Trades",
            "Total_Transaction_Cost",
        ]

        print(
            self.metrics[
                display_columns
            ].to_string(
                index=False
            )
        )

        print("=" * 100)

        # ----------------------------------------------------
        # SAVE METRICS
        # ----------------------------------------------------

        metrics_path = (
            RESULTS_DIRECTORY
            / "td3_validation_metrics.csv"
        )

        self.metrics.to_csv(
            metrics_path,
            index=False,
        )

        print(
            f"\nMetrics saved:"
            f"\n{metrics_path}"
        )

        # ----------------------------------------------------
        # SAVE HISTORIES
        # ----------------------------------------------------

        td3_history_path = (
            RESULTS_DIRECTORY
            / "td3_validation_history.csv"
        )

        buy_hold_history_path = (
            RESULTS_DIRECTORY
            / "buy_hold_validation_history.csv"
        )

        self.td3_history.to_csv(
            td3_history_path,
            index=False,
        )

        self.buy_hold_history.to_csv(
            buy_hold_history_path,
            index=False,
        )

        print(
            "\nDetailed histories saved."
        )

        return self.metrics

    # ========================================================
    # VISUALIZATION
    # ========================================================

    def plot_comparison(self):

        print("\n" + "=" * 80)
        print("GENERATING VALIDATION VISUALIZATIONS")
        print("=" * 80)

        if (
            self.td3_history is None
            or self.buy_hold_history is None
        ):

            raise RuntimeError(
                "Evaluate both strategies before plotting."
            )

        # ----------------------------------------------------
        # FIGURE
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(16, 20),
        )

        # ====================================================
        # 1. PORTFOLIO VALUE
        # ====================================================

        print(
            "\nPlotting portfolio value comparison..."
        )

        axes[0].plot(
            self.td3_history["Date"],
            self.td3_history["Portfolio_Value"],
            label="TD3",
        )

        axes[0].plot(
            self.buy_hold_history["Date"],
            self.buy_hold_history["Portfolio_Value"],
            label="Buy & Hold",
        )

        axes[0].axhline(
            self.initial_capital,
            linestyle="--",
            label="Initial Capital",
        )

        axes[0].set_title(
            "TD3 vs Buy & Hold - Portfolio Value"
        )

        axes[0].set_ylabel(
            "Portfolio Value (₹)"
        )

        axes[0].grid(
            True,
            alpha=0.3,
        )

        axes[0].legend()

        # ====================================================
        # 2. DRAWDOWN
        # ====================================================

        print(
            "\nPlotting drawdown comparison..."
        )

        axes[1].plot(
            self.td3_history["Date"],
            self.td3_history["Drawdown"],
            label="TD3 Drawdown",
        )

        axes[1].plot(
            self.buy_hold_history["Date"],
            self.buy_hold_history["Drawdown"],
            label="Buy & Hold Drawdown",
        )

        axes[1].set_title(
            "Drawdown Comparison"
        )

        axes[1].set_ylabel(
            "Drawdown"
        )

        axes[1].grid(
            True,
            alpha=0.3,
        )

        axes[1].legend()

        # ====================================================
        # 3. TD3 ALLOCATION
        # ====================================================

        print(
            "\nPlotting TD3 allocation..."
        )

        axes[2].plot(
            self.td3_history["Date"],
            self.td3_history["Position"],
            label="TD3 Allocation",
        )

        axes[2].set_title(
            "TD3 Portfolio Allocation Over Time"
        )

        axes[2].set_ylabel(
            "Investment Allocation"
        )

        axes[2].set_ylim(
            0,
            1,
        )

        axes[2].grid(
            True,
            alpha=0.3,
        )

        axes[2].legend()

        # ====================================================
        # 4. DAILY RETURNS
        # ====================================================

        print(
            "\nPlotting daily returns..."
        )

        axes[3].plot(
            self.td3_history["Date"],
            self.td3_history["Daily_Return"],
            label="TD3 Daily Return",
        )

        axes[3].plot(
            self.buy_hold_history["Date"],
            self.buy_hold_history["Daily_Return"],
            label="Buy & Hold Daily Return",
            alpha=0.7,
        )

        axes[3].set_title(
            "Daily Returns Comparison"
        )

        axes[3].set_ylabel(
            "Daily Return"
        )

        axes[3].set_xlabel(
            "Date"
        )

        axes[3].grid(
            True,
            alpha=0.3,
        )

        axes[3].legend()

        # ====================================================
        # LAYOUT
        # ====================================================

        plt.tight_layout()

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        save_path = (
            PLOTS_DIRECTORY
            / "td3_validation_comparison.png"
        )

        print(
            "\nSaving visualization..."
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"\nVisualization saved:"
            f"\n{save_path}"
        )

        plt.show()

        plt.close()

        print(
            "\nVisualization completed."
        )

    # ========================================================
    # RUN COMPLETE EVALUATION
    # ========================================================

    def run_evaluation(self):

        print("\n\n")

        print("=" * 100)

        print(
            "TRADE RL - TD3 MODEL EVALUATION"
        )

        print("=" * 100)

        # ----------------------------------------------------
        # ENVIRONMENTS
        # ----------------------------------------------------

        print(
            "\nPHASE 1: CREATE VALIDATION ENVIRONMENTS"
        )

        self.create_environments()

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        print(
            "\nPHASE 2: LOAD TRAINED MODEL"
        )

        self.load_model()

        # ----------------------------------------------------
        # TD3
        # ----------------------------------------------------

        print(
            "\nPHASE 3: EVALUATE TD3"
        )

        self.evaluate_td3()

        # ----------------------------------------------------
        # BUY & HOLD
        # ----------------------------------------------------

        print(
            "\nPHASE 4: EVALUATE BUY & HOLD"
        )

        self.evaluate_buy_and_hold()

        # ----------------------------------------------------
        # COMPARISON
        # ----------------------------------------------------

        print(
            "\nPHASE 5: COMPARE STRATEGIES"
        )

        self.compare_strategies()

        # ----------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------

        print(
            "\nPHASE 6: GENERATE VISUALIZATIONS"
        )

        self.plot_comparison()

        # ----------------------------------------------------
        # COMPLETION
        # ----------------------------------------------------

        print("\n")

        print("=" * 100)

        print(
            "TD3 VALIDATION EVALUATION COMPLETED"
        )

        print("=" * 100)

        print(
            f"\nMetrics:"
            f"\n{RESULTS_DIRECTORY / 'td3_validation_metrics.csv'}"
        )

        print(
            f"\nVisualization:"
            f"\n{PLOTS_DIRECTORY / 'td3_validation_comparison.png'}"
        )

        print("=" * 100)

        return self.metrics


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("\n")

    print("=" * 100)

    print(
        "TRADE RL - TD3 VALIDATION EVALUATION"
    )

    print("=" * 100)

    evaluator = TD3Evaluator()

    evaluator.run_evaluation()


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    main()