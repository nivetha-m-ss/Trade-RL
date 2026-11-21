"""
TD3 Agent for Trade RL

Uses Stable-Baselines3 TD3 implementation.

The agent receives market observations and produces
continuous target portfolio allocations.

Action:
    0.0 -> 100% Cash
    1.0 -> 100% Invested
"""

from pathlib import Path

import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIRECTORY = (
    PROJECT_ROOT / "models" / "trained"
)


# ============================================================
# TD3 AGENT
# ============================================================

class TD3Agent:

    def __init__(
        self,
        environment,
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
        seed=42,
        device="auto",
    ):

        print("\n" + "=" * 70)
        print("INITIALIZING TD3 AGENT")
        print("=" * 70)

        self.environment = environment

        self.seed = seed

        self.device = device

        # ----------------------------------------------------
        # ACTION DIMENSION
        # ----------------------------------------------------

        action_dimension = (
            environment.action_space.shape[0]
        )

        print(
            f"Action Dimension: {action_dimension}"
        )

        # ----------------------------------------------------
        # EXPLORATION NOISE
        # ----------------------------------------------------

        print("\nConfiguring exploration noise...")

        action_noise = NormalActionNoise(
            mean=np.zeros(
                action_dimension
            ),
            sigma=exploration_noise
            * np.ones(
                action_dimension
            ),
        )

        # ----------------------------------------------------
        # TD3 PARAMETERS
        # ----------------------------------------------------

        self.config = {

            "learning_rate": learning_rate,

            "buffer_size": buffer_size,

            "learning_starts": learning_starts,

            "batch_size": batch_size,

            "tau": tau,

            "gamma": gamma,

            "train_freq": train_freq,

            "gradient_steps": gradient_steps,

            "policy_delay": policy_delay,

            "target_policy_noise": target_policy_noise,

            "target_noise_clip": target_noise_clip,

            "exploration_noise": exploration_noise,

            "seed": seed,

            "device": device,
        }

        # ----------------------------------------------------
        # CREATE TD3 MODEL
        # ----------------------------------------------------

        print("\nCreating TD3 model...")

        self.model = TD3(

            policy="MlpPolicy",

            env=self.environment,

            learning_rate=learning_rate,

            buffer_size=buffer_size,

            learning_starts=learning_starts,

            batch_size=batch_size,

            tau=tau,

            gamma=gamma,

            train_freq=train_freq,

            gradient_steps=gradient_steps,

            policy_delay=policy_delay,

            target_policy_noise=target_policy_noise,

            target_noise_clip=target_noise_clip,

            action_noise=action_noise,

            seed=seed,

            device=device,

            verbose=1,
        )

        print("\nTD3 Agent initialized successfully.")

        print(
            f"Observation Shape: "
            f"{environment.observation_space.shape}"
        )

        print(
            f"Action Shape: "
            f"{environment.action_space.shape}"
        )

        print(
            f"Learning Rate: "
            f"{learning_rate}"
        )

        print(
            f"Batch Size: "
            f"{batch_size}"
        )

        print(
            f"Replay Buffer Size: "
            f"{buffer_size:,}"
        )

        print(
            f"Discount Factor: "
            f"{gamma}"
        )

        print(
            f"Exploration Noise: "
            f"{exploration_noise}"
        )

        print("=" * 70)

    # ========================================================
    # ACTION SELECTION
    # ========================================================

    def predict(
        self,
        observation,
        deterministic=False,
    ):

        action, state = self.model.predict(
            observation,
            deterministic=deterministic,
        )

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        action = np.clip(
            action,
            0.0,
            1.0,
        )

        return action, state

    # ========================================================
    # TRAINING
    # ========================================================

    def train(
        self,
        total_timesteps,
        callback=None,
    ):

        print("\n" + "=" * 70)
        print("STARTING TD3 TRAINING")
        print("=" * 70)

        print(
            f"Total Timesteps: "
            f"{total_timesteps:,}"
        )

        print("\nTraining started...")

        self.model.learn(

            total_timesteps=total_timesteps,

            callback=callback,

            progress_bar=True,
        )

        print("\nTraining completed.")

        print("=" * 70)

    # ========================================================
    # SAVE MODEL
    # ========================================================

    def save(
        self,
        model_name="td3_nifty50",
    ):

        print("\nSaving TD3 model...")

        MODEL_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = (
            MODEL_DIRECTORY / model_name
        )

        self.model.save(
            str(model_path)
        )

        print(
            f"Model saved successfully:"
            f"\n{model_path}"
        )

        return model_path

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load(
        self,
        model_path,
    ):

        print("\nLoading TD3 model...")

        model_path = Path(
            model_path
        )

        self.model = TD3.load(
            str(model_path),
            env=self.environment,
            device=self.device,
        )

        print(
            "TD3 model loaded successfully."
        )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    def get_model_info(self):

        print("\n" + "=" * 70)
        print("TD3 MODEL INFORMATION")
        print("=" * 70)

        print(
            f"Algorithm: TD3"
        )

        print(
            f"Policy: MlpPolicy"
        )

        print(
            f"Observation Space: "
            f"{self.environment.observation_space}"
        )

        print(
            f"Action Space: "
            f"{self.environment.action_space}"
        )

        print(
            f"Device: "
            f"{self.model.device}"
        )

        print("\nConfiguration:")

        for key, value in self.config.items():

            print(
                f"{key}: {value}"
            )

        print("=" * 70)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("\nRunning TD3 Agent Test...")

    from src.environment.trading_environment import (
        TradingEnvironment,
    )

    # --------------------------------------------------------
    # CREATE ENVIRONMENT
    # --------------------------------------------------------

    print("\nCreating trading environment...")

    env = TradingEnvironment()

    # --------------------------------------------------------
    # CREATE AGENT
    # --------------------------------------------------------

    print("\nCreating TD3 agent...")

    agent = TD3Agent(
        environment=env,
        seed=42,
    )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    agent.get_model_info()

    # --------------------------------------------------------
    # RESET ENVIRONMENT
    # --------------------------------------------------------

    print("\nResetting environment...")

    observation, info = env.reset(
        seed=42
    )

    print(
        f"Initial Portfolio Value: "
        f"₹{info['portfolio_value']:,.2f}"
    )

    # --------------------------------------------------------
    # TEST ACTION PREDICTION
    # --------------------------------------------------------

    print("\nTesting TD3 action prediction...")

    action, _ = agent.predict(
        observation,
        deterministic=True,
    )

    print(
        f"Predicted Action: {action}"
    )

    print(
        f"Target Allocation: "
        f"{action[0]:.2%}"
    )

    # --------------------------------------------------------
    # TEST ENVIRONMENT INTERACTION
    # --------------------------------------------------------

    print("\nTesting environment interaction...")

    for step in range(10):

        action, _ = agent.predict(
            observation,
            deterministic=False,
        )

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        print(
            f"\nStep: {step + 1}"
        )

        print(
            f"Action: {action[0]:.4f}"
        )

        print(
            f"Allocation: {info['position']:.2%}"
        )

        print(
            f"Reward: {reward:.6f}"
        )

        print(
            f"Portfolio Value: "
            f"₹{info['portfolio_value']:,.2f}"
        )

        print(
            f"Trade Type: "
            f"{info['trade_type']}"
        )

        if terminated or truncated:

            print(
                "\nEnvironment terminated."
            )

            break

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\nFinal Portfolio Information:")

    print(
        env.get_portfolio_info()
    )

    print("\nTD3 Agent test completed.")

    print("=" * 70)