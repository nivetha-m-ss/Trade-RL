"""
TD3 Trading Environment for Nifty 50

Action:
    Continuous target allocation between 0 and 1.

    0.0 = 100% Cash
    0.5 = 50% invested
    1.0 = 100% invested

Execution:
    Action is decided using the current day's features.
    Trading executes at the next day's Open.
    Portfolio is valued at the next day's Close.

Reward:
    Daily portfolio return.
"""

import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

from gymnasium import spaces
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "splits" / "train.csv"
)

INITIAL_CAPITAL = 100_000
TRANSACTION_COST = 0.001

FEATURE_COLUMNS = [
    "Daily_Return",
    "RSI_14",
    "MACD_Histogram_Pct",
    "Close_SMA20_Ratio",
    "Volatility_20",
]


# ============================================================
# TRADING ENVIRONMENT
# ============================================================

class TradingEnvironment(gym.Env):

    metadata = {
        "render_modes": ["human"],
    }

    def __init__(
        self,
        data_path=DEFAULT_DATA_PATH,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        render_mode=None,
    ):

        print("\n" + "=" * 70)
        print("INITIALIZING TD3 TRADING ENVIRONMENT")
        print("=" * 70)

        super().__init__()

        self.data_path = Path(data_path)
        self.initial_capital = float(initial_capital)
        self.transaction_cost = float(transaction_cost)
        self.render_mode = render_mode

        print(f"Data path        : {self.data_path}")
        print(f"Initial capital  : ₹{self.initial_capital:,.2f}")
        print(f"Transaction cost : {self.transaction_cost * 100:.2f}%")

        # ----------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------

        print("\nLoading market data...")

        self.data = self._load_data()

        print(f"Loaded rows      : {len(self.data):,}")
        print(f"Start date       : {self.data['Date'].iloc[0]}")
        print(f"End date         : {self.data['Date'].iloc[-1]}")

        # ----------------------------------------------------
        # FEATURE CONFIGURATION
        # ----------------------------------------------------

        self.feature_columns = FEATURE_COLUMNS.copy()

        self.observation_size = len(self.feature_columns) + 3

        # Additional state:
        # Position
        # Cash ratio
        # Portfolio return

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_size,),
            dtype=np.float32,
        )

        # Continuous target allocation
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # INITIAL STATE
        # ----------------------------------------------------

        self.current_step = 0

        self.cash = self.initial_capital
        self.shares = 0.0

        self.portfolio_value = self.initial_capital
        self.previous_portfolio_value = self.initial_capital

        self.position = 0.0

        self.total_transaction_cost = 0.0
        self.total_trades = 0

        self.portfolio_history = []
        self.reward_history = []
        self.action_history = []
        self.trade_history = []

        print("\nEnvironment initialized successfully.")
        print(f"Observation shape: {self.observation_space.shape}")
        print(f"Action space     : {self.action_space}")
        print("=" * 70)

    # ========================================================
    # DATA LOADING
    # ========================================================

    def _load_data(self):

        print("Reading CSV file...")

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}"
            )

        df = pd.read_csv(self.data_path)

        print(f"Raw dataset rows: {len(df):,}")

        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        if "Date" not in df.columns:
            raise ValueError("Dataset must contain Date column.")

        df["Date"] = pd.to_datetime(df["Date"])

        # ----------------------------------------------------
        # REQUIRED MARKET DATA
        # ----------------------------------------------------

        required_market_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        missing_columns = [
            column
            for column in required_market_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing market columns: {missing_columns}"
            )

        # ----------------------------------------------------
        # REQUIRED FEATURES
        # ----------------------------------------------------

        missing_features = [
            column
            for column in FEATURE_COLUMNS
            if column not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing feature columns: {missing_features}"
            )

        # ----------------------------------------------------
        # CLEAN DATA
        # ----------------------------------------------------

        print("Cleaning dataset...")

        df = df.dropna(
            subset=[
                "Date",
                "Open",
                "Close",
                *FEATURE_COLUMNS,
            ]
        )

        df = df.sort_values("Date")

        df = df.drop_duplicates(
            subset=["Date"],
            keep="first",
        )

        df = df.reset_index(drop=True)

        # ----------------------------------------------------
        # NUMERIC VALIDATION
        # ----------------------------------------------------

        numeric_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            *FEATURE_COLUMNS,
        ]

        for column in numeric_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        df = df.dropna(
            subset=numeric_columns
        )

        df = df.reset_index(drop=True)

        if len(df) < 2:
            raise ValueError(
                "Not enough data rows to run the environment."
            )

        print("Dataset cleaning completed.")
        print(f"Final rows: {len(df):,}")

        return df

    # ========================================================
    # OBSERVATION
    # ========================================================

    def _get_observation(self):

        row = self.data.iloc[self.current_step]

        market_features = np.array(
            [
                row[column]
                for column in self.feature_columns
            ],
            dtype=np.float32,
        )

        current_price = float(row["Close"])

        holdings_value = self.shares * current_price

        portfolio_value = self.cash + holdings_value

        cash_ratio = (
            self.cash / portfolio_value
            if portfolio_value > 0
            else 0.0
        )

        portfolio_return = (
            portfolio_value / self.initial_capital
        ) - 1.0

        position = (
            holdings_value / portfolio_value
            if portfolio_value > 0
            else 0.0
        )

        observation = np.concatenate(
            [
                market_features,
                np.array(
                    [
                        position,
                        cash_ratio,
                        portfolio_return,
                    ],
                    dtype=np.float32,
                ),
            ]
        )

        return observation.astype(np.float32)

    # ========================================================
    # PORTFOLIO VALUATION
    # ========================================================

    def _calculate_portfolio_value(self, price):

        holdings_value = self.shares * price

        return self.cash + holdings_value

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):

        print("\n" + "-" * 60)
        print("RESETTING TRADING ENVIRONMENT")
        print("-" * 60)

        super().reset(seed=seed)

        self.current_step = 0

        self.cash = self.initial_capital
        self.shares = 0.0

        self.portfolio_value = self.initial_capital
        self.previous_portfolio_value = self.initial_capital

        self.position = 0.0

        self.total_transaction_cost = 0.0
        self.total_trades = 0

        self.portfolio_history = []
        self.reward_history = []
        self.action_history = []
        self.trade_history = []

        initial_price = float(
            self.data.iloc[0]["Close"]
        )

        self.portfolio_history.append(
            {
                "Date": self.data.iloc[0]["Date"],
                "Portfolio_Value": self.initial_capital,
                "Cash": self.cash,
                "Shares": self.shares,
                "Holdings_Value": 0.0,
                "Position": 0.0,
                "Reward": 0.0,
            }
        )

        observation = self._get_observation()

        info = {
            "date": self.data.iloc[0]["Date"],
            "price": initial_price,
            "cash": self.cash,
            "shares": self.shares,
            "portfolio_value": self.portfolio_value,
            "position": self.position,
        }

        print("Environment reset completed.")
        print(f"Initial capital: ₹{self.initial_capital:,.2f}")
        print(f"Initial price : ₹{initial_price:,.2f}")

        return observation, info

    # ========================================================
    # ACTION EXECUTION
    # ========================================================

    def _execute_action(
        self,
        target_allocation,
        execution_price,
    ):

        target_allocation = float(
            np.clip(
                target_allocation,
                0.0,
                1.0,
            )
        )

        current_portfolio_value = (
            self.cash
            + self.shares * execution_price
        )

        target_holdings_value = (
            current_portfolio_value
            * target_allocation
        )

        target_shares = (
            target_holdings_value
            / execution_price
        )

        shares_to_trade = (
            target_shares - self.shares
        )

        transaction_cost = 0.0

        trade_type = "HOLD"

        # ----------------------------------------------------
        # BUY
        # ----------------------------------------------------

        if shares_to_trade > 0:

            trade_value = (
                shares_to_trade
                * execution_price
            )

            transaction_cost = (
                trade_value
                * self.transaction_cost
            )

            total_cost = (
                trade_value
                + transaction_cost
            )

            # Prevent negative cash
            if total_cost > self.cash:

                affordable_value = (
                    self.cash
                    / (1 + self.transaction_cost)
                )

                affordable_shares = (
                    affordable_value
                    / execution_price
                )

                shares_to_trade = affordable_shares

                trade_value = (
                    shares_to_trade
                    * execution_price
                )

                transaction_cost = (
                    trade_value
                    * self.transaction_cost
                )

                total_cost = (
                    trade_value
                    + transaction_cost
                )

            self.cash -= total_cost

            self.shares += shares_to_trade

            trade_type = "BUY"

        # ----------------------------------------------------
        # SELL
        # ----------------------------------------------------

        elif shares_to_trade < 0:

            shares_to_sell = abs(
                shares_to_trade
            )

            shares_to_sell = min(
                shares_to_sell,
                self.shares,
            )

            trade_value = (
                shares_to_sell
                * execution_price
            )

            transaction_cost = (
                trade_value
                * self.transaction_cost
            )

            net_proceeds = (
                trade_value
                - transaction_cost
            )

            self.cash += net_proceeds

            self.shares -= shares_to_sell

            trade_type = "SELL"

            shares_to_trade = -shares_to_sell

        # ----------------------------------------------------
        # UPDATE
        # ----------------------------------------------------

        if trade_type != "HOLD":

            self.total_trades += 1

            self.total_transaction_cost += (
                transaction_cost
            )

            trade_info = {
                "Date": self.data.iloc[
                    self.current_step + 1
                ]["Date"],
                "Trade_Type": trade_type,
                "Execution_Price": execution_price,
                "Shares": abs(shares_to_trade),
                "Trade_Value": abs(
                    shares_to_trade
                    * execution_price
                ),
                "Transaction_Cost": transaction_cost,
                "Target_Allocation": target_allocation,
            }

            self.trade_history.append(
                trade_info
            )

        self.position = (
            self.shares
            * execution_price
            / current_portfolio_value
            if current_portfolio_value > 0
            else 0.0
        )

        return {
            "trade_type": trade_type,
            "shares_traded": abs(shares_to_trade),
            "transaction_cost": transaction_cost,
            "target_allocation": target_allocation,
        }

    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        # ----------------------------------------------------
        # ACTION VALIDATION
        # ----------------------------------------------------

        target_allocation = float(
            np.asarray(action).flatten()[0]
        )

        target_allocation = float(
            np.clip(
                target_allocation,
                0.0,
                1.0,
            )
        )

        # ----------------------------------------------------
        # TERMINATION CHECK
        # ----------------------------------------------------

        if self.current_step >= len(self.data) - 1:

            observation = self._get_observation()

            return (
                observation,
                0.0,
                True,
                False,
                {
                    "portfolio_value": self.portfolio_value,
                    "message": "Dataset exhausted.",
                },
            )

        # ----------------------------------------------------
        # CURRENT DATA
        # ----------------------------------------------------

        current_row = self.data.iloc[
            self.current_step
        ]

        next_row = self.data.iloc[
            self.current_step + 1
        ]

        current_date = current_row["Date"]

        execution_date = next_row["Date"]

        execution_price = float(
            next_row["Open"]
        )

        market_close = float(
            next_row["Close"]
        )

        previous_portfolio_value = (
            self._calculate_portfolio_value(
                float(current_row["Close"])
            )
        )

        # ----------------------------------------------------
        # EXECUTE ACTION AT NEXT OPEN
        # ----------------------------------------------------

        trade_info = self._execute_action(
            target_allocation=target_allocation,
            execution_price=execution_price,
        )

        # ----------------------------------------------------
        # MOVE TO NEXT DAY
        # ----------------------------------------------------

        self.current_step += 1

        # ----------------------------------------------------
        # VALUE PORTFOLIO AT NEXT CLOSE
        # ----------------------------------------------------

        self.cash = max(
            self.cash,
            0.0,
        )

        self.portfolio_value = (
            self.cash
            + self.shares * market_close
        )

        # ----------------------------------------------------
        # REWARD
        # ----------------------------------------------------

        reward = (
            self.portfolio_value
            - previous_portfolio_value
        ) / previous_portfolio_value

        self.previous_portfolio_value = (
            previous_portfolio_value
        )

        # ----------------------------------------------------
        # UPDATE POSITION
        # ----------------------------------------------------

        holdings_value = (
            self.shares
            * market_close
        )

        self.position = (
            holdings_value
            / self.portfolio_value
            if self.portfolio_value > 0
            else 0.0
        )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        self.reward_history.append(
            reward
        )

        self.action_history.append(
            target_allocation
        )

        self.portfolio_history.append(
            {
                "Date": execution_date,
                "Portfolio_Value": self.portfolio_value,
                "Cash": self.cash,
                "Shares": self.shares,
                "Holdings_Value": holdings_value,
                "Position": self.position,
                "Reward": reward,
                "Action": target_allocation,
                "Trade_Type": trade_info["trade_type"],
                "Execution_Price": execution_price,
                "Market_Close": market_close,
                "Transaction_Cost": trade_info[
                    "transaction_cost"
                ],
            }
        )

        # ----------------------------------------------------
        # TERMINATION
        # ----------------------------------------------------

        terminated = (
            self.current_step
            >= len(self.data) - 1
        )

        truncated = False

        # ----------------------------------------------------
        # NEXT OBSERVATION
        # ----------------------------------------------------

        observation = self._get_observation()

        # ----------------------------------------------------
        # INFO
        # ----------------------------------------------------

        info = {
            "date": execution_date,
            "action": target_allocation,
            "trade_type": trade_info["trade_type"],
            "execution_price": execution_price,
            "market_close": market_close,
            "cash": self.cash,
            "shares": self.shares,
            "holdings_value": holdings_value,
            "position": self.position,
            "portfolio_value": self.portfolio_value,
            "previous_portfolio_value": previous_portfolio_value,
            "reward": reward,
            "transaction_cost": trade_info[
                "transaction_cost"
            ],
            "total_transaction_cost": self.total_transaction_cost,
            "total_trades": self.total_trades,
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # ========================================================
    # PORTFOLIO INFORMATION
    # ========================================================

    def get_portfolio_info(self):

        current_row = self.data.iloc[
            self.current_step
        ]

        current_price = float(
            current_row["Close"]
        )

        holdings_value = (
            self.shares
            * current_price
        )

        portfolio_value = (
            self.cash
            + holdings_value
        )

        portfolio_return = (
            portfolio_value
            / self.initial_capital
        ) - 1

        return {
            "Date": current_row["Date"],
            "Cash": self.cash,
            "Shares": self.shares,
            "Holdings_Value": holdings_value,
            "Portfolio_Value": portfolio_value,
            "Position": self.position,
            "Portfolio_Return": portfolio_return,
            "Total_Transaction_Cost": self.total_transaction_cost,
            "Total_Trades": self.total_trades,
        }

    # ========================================================
    # RENDER
    # ========================================================

    def render(self):

        portfolio_info = (
            self.get_portfolio_info()
        )

        print(
            f"\nDate: {portfolio_info['Date']}"
        )

        print(
            f"Portfolio Value: ₹"
            f"{portfolio_info['Portfolio_Value']:,.2f}"
        )

        print(
            f"Cash: ₹"
            f"{portfolio_info['Cash']:,.2f}"
        )

        print(
            f"Shares: "
            f"{portfolio_info['Shares']:.4f}"
        )

        print(
            f"Allocation: "
            f"{portfolio_info['Position']:.2%}"
        )

        print(
            f"Return: "
            f"{portfolio_info['Portfolio_Return']:.2%}"
        )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    def plot_results(
        self,
        save_path=None,
    ):

        print("\nGenerating trading environment visualization...")

        if not self.portfolio_history:

            print(
                "No portfolio history available."
            )

            return

        history = pd.DataFrame(
            self.portfolio_history
        )

        history["Date"] = pd.to_datetime(
            history["Date"]
        )

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(15, 12),
        )

        # ----------------------------------------------------
        # PORTFOLIO VALUE
        # ----------------------------------------------------

        axes[0].plot(
            history["Date"],
            history["Portfolio_Value"],
            label="Portfolio Value",
        )

        axes[0].axhline(
            self.initial_capital,
            linestyle="--",
            label="Initial Capital",
        )

        axes[0].set_title(
            "Portfolio Value"
        )

        axes[0].set_ylabel(
            "Portfolio Value (₹)"
        )

        axes[0].grid(
            True,
            alpha=0.3,
        )

        axes[0].legend()

        # ----------------------------------------------------
        # ALLOCATION
        # ----------------------------------------------------

        if "Position" in history.columns:

            axes[1].plot(
                history["Date"],
                history["Position"],
                label="Portfolio Allocation",
            )

            axes[1].set_title(
                "Portfolio Allocation"
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

        if "Reward" in history.columns:

            axes[2].plot(
                history["Date"],
                history["Reward"],
                label="Daily Reward",
            )

            axes[2].set_title(
                "Daily Portfolio Return / Reward"
            )

            axes[2].set_ylabel(
                "Reward"
            )

            axes[2].grid(
                True,
                alpha=0.3,
            )

            axes[2].legend()

        plt.tight_layout()

        if save_path is None:

            save_path = (
                PROJECT_ROOT
                / "results"
                / "plots"
                / "trading_environment_test.png"
            )

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Visualization saved: {save_path}"
        )

        plt.show()

        plt.close()

    # ========================================================
    # ENVIRONMENT TEST
    # ========================================================

    def test_environment(self):

        print("\n" + "=" * 70)
        print("TRADING ENVIRONMENT TEST")
        print("=" * 70)

        print("\nTesting reset...")

        observation, info = self.reset(
            seed=42
        )

        print("\nInitial observation:")
        print(observation)

        print("\nInitial info:")
        print(info)

        print("\nTesting continuous actions...")

        test_actions = [
            1.0,    # 100% invested
            0.5,    # 50% invested
            0.0,    # 100% cash
            0.75,   # 75% invested
            0.25,   # 25% invested
        ]

        for i, action in enumerate(
            test_actions,
            start=1,
        ):

            print(
                f"\nTest Action {i}: "
                f"{action:.2f}"
            )

            result = self.step(
                np.array(
                    [action],
                    dtype=np.float32,
                )
            )

            observation, reward, terminated, truncated, info = result

            print(
                f"Trade Type       : "
                f"{info['trade_type']}"
            )

            print(
                f"Execution Price  : ₹"
                f"{info['execution_price']:,.2f}"
            )

            print(
                f"Portfolio Value  : ₹"
                f"{info['portfolio_value']:,.2f}"
            )

            print(
                f"Cash             : ₹"
                f"{info['cash']:,.2f}"
            )

            print(
                f"Allocation       : "
                f"{info['position']:.2%}"
            )

            print(
                f"Reward           : "
                f"{reward:.6f}"
            )

            print(
                f"Transaction Cost : ₹"
                f"{info['transaction_cost']:,.2f}"
            )

            if terminated or truncated:

                print(
                    "\nEnvironment terminated."
                )

                break

        print("\nFinal Portfolio Information:")

        print(
            self.get_portfolio_info()
        )

        print("\nGenerating visualization...")

        self.plot_results()

        print("\nEnvironment test completed.")

        print("=" * 70)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("\nRunning Trading Environment Test...")

    env = TradingEnvironment()

    env.test_environment()