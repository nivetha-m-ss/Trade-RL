"""
TD3 Training Pipeline for Trade RL

Responsibilities:
1. Initialize training environment
2. Initialize TD3 agent
3. Train TD3 model
4. Track training progress
5. Save checkpoints
6. Save final model
7. Evaluate trained model on training data
8. Generate training visualizations
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# IMPORTS
# ============================================================

from src.environment.trading_environment import (
    TradingEnvironment,
)

from src.agent.td3_agent import (
    TD3Agent,
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "train.csv"
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "trained"
)

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
)

PLOTS_DIRECTORY = (
    RESULTS_DIRECTORY
    / "plots"
)

LOGS_DIRECTORY = (
    RESULTS_DIRECTORY
    / "logs"
)

TOTAL_TIMESTEPS = 50_000

INITIAL_CAPITAL = 100_000

TRANSACTION_COST = 0.001

SEED = 42

CHECKPOINT_FREQUENCY = 5_000


# ============================================================
# TRAINING CALLBACK
# ============================================================

class TD3TrainingCallback(BaseCallback):

    def __init__(
        self,
        checkpoint_frequency=5_000,
        checkpoint_directory=None,
        verbose=1,
    ):

        super().__init__(
            verbose=verbose
        )

        self.checkpoint_frequency = (
            checkpoint_frequency
        )

        self.checkpoint_directory = (
            Path(checkpoint_directory)
            if checkpoint_directory
            else MODEL_DIRECTORY / "checkpoints"
        )

        self.episode_rewards = []

        self.episode_lengths = []

        self.timesteps = []

        self.last_checkpoint = 0

    # ========================================================
    # CALLBACK INITIALIZATION
    # ========================================================

    def _on_training_start(self):

        print("\n" + "=" * 70)
        print("TD3 TRAINING CALLBACK INITIALIZED")
        print("=" * 70)

        print(
            f"Checkpoint frequency: "
            f"{self.checkpoint_frequency:,}"
        )

        self.checkpoint_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # CALLBACK STEP
    # ========================================================

    def _on_step(self):

        # ----------------------------------------------------
        # EPISODE INFORMATION
        # ----------------------------------------------------

        infos = self.locals.get(
            "infos",
            [],
        )

        for info in infos:

            if "episode" in info:

                episode_reward = (
                    info["episode"]["r"]
                )

                episode_length = (
                    info["episode"]["l"]
                )

                self.episode_rewards.append(
                    episode_reward
                )

                self.episode_lengths.append(
                    episode_length
                )

                self.timesteps.append(
                    self.num_timesteps
                )

                if self.verbose:

                    print(
                        "\nEpisode Completed"
                    )

                    print(
                        f"Episode Reward: "
                        f"{episode_reward:.6f}"
                    )

                    print(
                        f"Episode Length: "
                        f"{episode_length}"
                    )

                    print(
                        f"Total Timesteps: "
                        f"{self.num_timesteps:,}"
                    )

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        if (
            self.num_timesteps
            - self.last_checkpoint
            >= self.checkpoint_frequency
        ):

            checkpoint_path = (
                self.checkpoint_directory
                / f"td3_checkpoint_{self.num_timesteps}"
            )

            print("\nSaving checkpoint...")

            self.model.save(
                str(checkpoint_path)
            )

            print(
                f"Checkpoint saved: "
                f"{checkpoint_path}"
            )

            self.last_checkpoint = (
                self.num_timesteps
            )

        # ----------------------------------------------------
        # CONTINUE TRAINING
        # ----------------------------------------------------

        return True

    # ========================================================
    # CALLBACK END
    # ========================================================

    def _on_training_end(self):

        print("\n" + "=" * 70)
        print("TD3 TRAINING CALLBACK COMPLETED")
        print("=" * 70)

        print(
            f"Total Episodes: "
            f"{len(self.episode_rewards)}"
        )

        print(
            f"Total Timesteps: "
            f"{self.num_timesteps:,}"
        )


# ============================================================
# TD3 TRAINER
# ============================================================

class TD3Trainer:

    def __init__(
        self,
        total_timesteps=TOTAL_TIMESTEPS,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost=TRANSACTION_COST,
        seed=SEED,
    ):

        print("\n" + "=" * 80)
        print("INITIALIZING TD3 TRAINER")
        print("=" * 80)

        self.total_timesteps = (
            total_timesteps
        )

        self.initial_capital = (
            initial_capital
        )

        self.transaction_cost = (
            transaction_cost
        )

        self.seed = seed

        self.environment = None

        self.agent = None

        self.callback = None

        # ----------------------------------------------------
        # CREATE DIRECTORIES
        # ----------------------------------------------------

        print("\nCreating project directories...")

        MODEL_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        PLOTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        LOGS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("Directories created successfully.")

        print("\nTD3 Trainer initialized.")

        print(
            f"Total Timesteps: "
            f"{self.total_timesteps:,}"
        )

        print("=" * 80)

    # ========================================================
    # CREATE ENVIRONMENT
    # ========================================================

    def create_environment(self):

        print("\n" + "=" * 70)
        print("CREATING TRAINING ENVIRONMENT")
        print("=" * 70)

        print(
            f"Training Data: "
            f"{TRAIN_DATA_PATH}"
        )

        self.environment = TradingEnvironment(

            data_path=TRAIN_DATA_PATH,

            initial_capital=self.initial_capital,

            transaction_cost=self.transaction_cost,
        )

        print("\nWrapping environment with Monitor...")

        self.environment = Monitor(
            self.environment
        )

        print("Training environment created successfully.")

        return self.environment

    # ========================================================
    # CREATE AGENT
    # ========================================================

    def create_agent(self):

        print("\n" + "=" * 70)
        print("CREATING TD3 AGENT")
        print("=" * 70)

        if self.environment is None:

            raise RuntimeError(
                "Environment must be created before agent."
            )

        self.agent = TD3Agent(

            environment=self.environment,

            learning_rate=1e-3,

            buffer_size=100_000,

            learning_starts=1_000,

            batch_size=256,

            tau=0.005,

            gamma=0.99,

            train_freq=1,

            gradient_steps=1,

            policy_delay=2,

            target_policy_noise=0.2,

            target_noise_clip=0.5,

            exploration_noise=0.1,

            seed=self.seed,

            device="auto",
        )

        print("TD3 Agent created successfully.")

        return self.agent

    # ========================================================
    # CREATE CALLBACK
    # ========================================================

    def create_callback(self):

        print("\nCreating training callback...")

        self.callback = TD3TrainingCallback(

            checkpoint_frequency=CHECKPOINT_FREQUENCY,

            checkpoint_directory=(
                MODEL_DIRECTORY
                / "checkpoints"
            ),

            verbose=1,
        )

        print("Training callback created.")

        return self.callback

    # ========================================================
    # TRAIN
    # ========================================================

    def train(self):

        print("\n" + "=" * 80)
        print("STARTING TD3 TRAINING")
        print("=" * 80)

        print(
            f"Training Timesteps: "
            f"{self.total_timesteps:,}"
        )

        # ----------------------------------------------------
        # ENVIRONMENT
        # ----------------------------------------------------

        if self.environment is None:

            print("\nEnvironment not initialized.")

            self.create_environment()

        # ----------------------------------------------------
        # AGENT
        # ----------------------------------------------------

        if self.agent is None:

            print("\nAgent not initialized.")

            self.create_agent()

        # ----------------------------------------------------
        # CALLBACK
        # ----------------------------------------------------

        if self.callback is None:

            self.create_callback()

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        print("\nStarting model training...")

        print(
            "This operation may take some time."
        )

        self.agent.train(

            total_timesteps=self.total_timesteps,

            callback=self.callback,
        )

        print("\nTraining completed successfully.")

        # ----------------------------------------------------
        # SAVE FINAL MODEL
        # ----------------------------------------------------

        print("\nSaving final model...")

        final_model_path = self.agent.save(
            model_name="td3_nifty50_final"
        )

        print(
            f"\nFinal Model Path:"
            f"\n{final_model_path}"
        )

        # ----------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------

        print("\nGenerating training visualizations...")

        self.plot_training_results()

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return self.agent

    # ========================================================
    # TRAINING VISUALIZATION
    # ========================================================

    def plot_training_results(self):

        print("\nGenerating training reward plot...")

        if self.callback is None:

            print(
                "Training callback unavailable."
            )

            return

        if not self.callback.episode_rewards:

            print(
                "No completed episodes available for plotting."
            )

            return

        rewards = np.array(
            self.callback.episode_rewards
        )

        episodes = np.arange(
            1,
            len(rewards) + 1,
        )

        # ----------------------------------------------------
        # FIGURE
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(14, 10),
        )

        # ----------------------------------------------------
        # EPISODE REWARD
        # ----------------------------------------------------

        axes[0].plot(
            episodes,
            rewards,
            label="Episode Reward",
        )

        axes[0].set_title(
            "TD3 Training Episode Rewards"
        )

        axes[0].set_xlabel(
            "Episode"
        )

        axes[0].set_ylabel(
            "Episode Reward"
        )

        axes[0].grid(
            True,
            alpha=0.3,
        )

        axes[0].legend()

        # ----------------------------------------------------
        # MOVING AVERAGE
        # ----------------------------------------------------

        window_size = min(
            10,
            len(rewards),
        )

        if window_size > 1:

            moving_average = np.convolve(

                rewards,

                np.ones(
                    window_size
                ) / window_size,

                mode="valid",
            )

            axes[1].plot(
                np.arange(
                    window_size,
                    len(rewards) + 1,
                ),

                moving_average,

                label=f"{window_size}-Episode Moving Average",
            )

        axes[1].set_title(
            "TD3 Training Reward Moving Average"
        )

        axes[1].set_xlabel(
            "Episode"
        )

        axes[1].set_ylabel(
            "Average Reward"
        )

        axes[1].grid(
            True,
            alpha=0.3,
        )

        axes[1].legend()

        plt.tight_layout()

        save_path = (
            PLOTS_DIRECTORY
            / "td3_training_rewards.png"
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Training visualization saved:"
            f"\n{save_path}"
        )

        plt.show()

        plt.close()

    # ========================================================
    # DETERMINISTIC EVALUATION
    # ========================================================

    def evaluate_trained_agent(self):

        print("\n" + "=" * 80)
        print("EVALUATING TRAINED TD3 AGENT")
        print("=" * 80)

        if self.agent is None:

            raise RuntimeError(
                "TD3 Agent has not been trained."
            )

        # ----------------------------------------------------
        # USE RAW ENVIRONMENT
        # ----------------------------------------------------

        if isinstance(
            self.environment,
            Monitor,
        ):

            evaluation_environment = (
                self.environment.env
            )

        else:

            evaluation_environment = (
                self.environment
            )

        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------

        print("\nResetting environment...")

        observation, info = (
            evaluation_environment.reset(
                seed=self.seed
            )
        )

        print(
            f"Initial Portfolio Value: ₹"
            f"{info['portfolio_value']:,.2f}"
        )

        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        print("\nStarting deterministic evaluation...")

        step = 0

        while True:

            action, _ = self.agent.predict(

                observation,

                deterministic=True,
            )

            observation, reward, terminated, truncated, info = (
                evaluation_environment.step(
                    action
                )
            )

            step += 1

            if step % 100 == 0:

                print(
                    f"Evaluation Step: {step}"
                )

                print(
                    f"Portfolio Value: ₹"
                    f"{info['portfolio_value']:,.2f}"
                )

                print(
                    f"Allocation: "
                    f"{info['position']:.2%}"
                )

                print(
                    f"Reward: {reward:.6f}"
                )

            if terminated or truncated:

                break

        # ----------------------------------------------------
        # FINAL RESULTS
        # ----------------------------------------------------

        portfolio_info = (
            evaluation_environment.get_portfolio_info()
        )

        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)

        print(
            f"Initial Capital: ₹"
            f"{self.initial_capital:,.2f}"
        )

        print(
            f"Final Portfolio Value: ₹"
            f"{portfolio_info['Portfolio_Value']:,.2f}"
        )

        print(
            f"Total Return: "
            f"{portfolio_info['Portfolio_Return']:.2%}"
        )

        print(
            f"Total Trades: "
            f"{portfolio_info['Total_Trades']}"
        )

        print(
            f"Total Transaction Cost: ₹"
            f"{portfolio_info['Total_Transaction_Cost']:,.2f}"
        )

        print("=" * 80)

        # ----------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------

        print("\nGenerating evaluation visualization...")

        evaluation_environment.plot_results(
            save_path=(
                PLOTS_DIRECTORY
                / "td3_training_evaluation.png"
            )
        )

        return portfolio_info


# ============================================================
# MAIN TRAINING EXECUTION
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("TRADE RL - TD3 TRAINING")
    print("=" * 80)

    # --------------------------------------------------------
    # INITIALIZE TRAINER
    # --------------------------------------------------------

    trainer = TD3Trainer(

        total_timesteps=TOTAL_TIMESTEPS,

        initial_capital=INITIAL_CAPITAL,

        transaction_cost=TRANSACTION_COST,

        seed=SEED,
    )

    # --------------------------------------------------------
    # CREATE ENVIRONMENT
    # --------------------------------------------------------

    trainer.create_environment()

    # --------------------------------------------------------
    # CREATE AGENT
    # --------------------------------------------------------

    trainer.create_agent()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    trainer.train()

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    trainer.evaluate_trained_agent()

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("TD3 TRAINING PIPELINE COMPLETED")
    print("=" * 80)


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    main()