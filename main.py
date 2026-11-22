"""
Trade RL - Main Entry Point

Current functionality:
1. Validate project configuration
2. Initialize Trading Environment
3. Test Gymnasium compatibility
4. Test continuous action space
5. Run sample trading actions
6. Generate environment visualization
"""
#traing for git 

import sys
from pathlib import Path

import numpy as np

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# IMPORTS
# ============================================================

from src.environment.trading_environment import TradingEnvironment


# ============================================================
# CONFIGURATION
# ============================================================

INITIAL_CAPITAL = 100_000

TRANSACTION_COST = 0.001


# ============================================================
# PROJECT INFORMATION
# ============================================================

def display_project_information():

    print("\n")
    print("=" * 80)
    print("TRADE RL")
    print("=" * 80)

    print("Project          : Reinforcement Learning Trading System")
    print("Market           : Indian Stock Market")
    print("Index            : Nifty 50")
    print("Algorithm        : TD3")
    print("Action Space     : Continuous")
    print("Trading Strategy : Long-only")
    print("Initial Capital  : ₹100,000")
    print("Transaction Cost : 0.1%")

    print("=" * 80)


# ============================================================
# ENVIRONMENT VALIDATION
# ============================================================

def validate_environment(env):

    print("\n" + "=" * 80)
    print("ENVIRONMENT VALIDATION")
    print("=" * 80)

    # --------------------------------------------------------
    # OBSERVATION SPACE
    # --------------------------------------------------------

    print("\nChecking observation space...")

    print(
        f"Observation Space: {env.observation_space}"
    )

    print(
        f"Observation Shape: {env.observation_space.shape}"
    )

    # --------------------------------------------------------
    # ACTION SPACE
    # --------------------------------------------------------

    print("\nChecking action space...")

    print(
        f"Action Space: {env.action_space}"
    )

    print(
        f"Action Shape: {env.action_space.shape}"
    )

    print(
        f"Action Minimum: {env.action_space.low}"
    )

    print(
        f"Action Maximum: {env.action_space.high}"
    )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    print("\nTesting environment reset...")

    observation, info = env.reset(
        seed=42
    )

    print(
        f"Observation shape: {observation.shape}"
    )

    print(
        f"Observation dtype: {observation.dtype}"
    )

    print(
        f"Initial portfolio: ₹"
        f"{info['portfolio_value']:,.2f}"
    )

    assert observation.shape == env.observation_space.shape

    assert env.observation_space.contains(
        observation
    )

    print("Reset validation passed.")

    # --------------------------------------------------------
    # ACTION VALIDATION
    # --------------------------------------------------------

    print("\nTesting continuous actions...")

    test_actions = [
        np.array([0.0], dtype=np.float32),
        np.array([0.25], dtype=np.float32),
        np.array([0.50], dtype=np.float32),
        np.array([0.75], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    ]

    for action in test_actions:

        assert env.action_space.contains(
            action
        )

        print(
            f"Valid Action: {action[0]:.2f}"
        )

    print("Action validation passed.")

    # --------------------------------------------------------
    # STEP VALIDATION
    # --------------------------------------------------------

    print("\nTesting environment step...")

    observation, reward, terminated, truncated, info = env.step(
        np.array(
            [1.0],
            dtype=np.float32,
        )
    )

    print(
        f"Next observation shape: {observation.shape}"
    )

    print(
        f"Reward: {reward:.6f}"
    )

    print(
        f"Terminated: {terminated}"
    )

    print(
        f"Truncated: {truncated}"
    )

    print(
        f"Portfolio Value: ₹"
        f"{info['portfolio_value']:,.2f}"
    )

    assert observation.shape == env.observation_space.shape

    assert isinstance(
        reward,
        float,
    )

    print("Step validation passed.")

    print("\nEnvironment validation completed.")

    print("=" * 80)


# ============================================================
# ENVIRONMENT TEST
# ============================================================

def run_environment_test():

    print("\n")
    print("=" * 80)
    print("RUNNING TRADING ENVIRONMENT TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # INITIALIZE ENVIRONMENT
    # --------------------------------------------------------

    print("\nInitializing Trading Environment...")

    env = TradingEnvironment(
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
    )

    print("Environment initialization completed.")

    # --------------------------------------------------------
    # VALIDATE ENVIRONMENT
    # --------------------------------------------------------

    validate_environment(
        env
    )

    # --------------------------------------------------------
    # FULL ENVIRONMENT TEST
    # --------------------------------------------------------

    print("\nRunning full environment test...")

    env.test_environment()

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("ENVIRONMENT TEST SUMMARY")
    print("=" * 80)

    portfolio_info = env.get_portfolio_info()

    print(
        f"Final Portfolio Value: ₹"
        f"{portfolio_info['Portfolio_Value']:,.2f}"
    )

    print(
        f"Final Portfolio Return: "
        f"{portfolio_info['Portfolio_Return']:.2%}"
    )

    print(
        f"Total Transaction Cost: ₹"
        f"{portfolio_info['Total_Transaction_Cost']:,.2f}"
    )

    print(
        f"Total Trades: "
        f"{portfolio_info['Total_Trades']}"
    )

    print("=" * 80)

    print("\nTrading Environment is ready for TD3 integration.")


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # PROJECT INFORMATION
        # ----------------------------------------------------

        display_project_information()

        # ----------------------------------------------------
        # RUN ENVIRONMENT TEST
        # ----------------------------------------------------

        run_environment_test()

        print("\n")
        print("=" * 80)
        print("PROJECT EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 80)

    except Exception as error:

        print("\n")
        print("=" * 80)
        print("PROJECT EXECUTION FAILED")
        print("=" * 80)

        print(f"Error: {error}")

        raise


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    main()