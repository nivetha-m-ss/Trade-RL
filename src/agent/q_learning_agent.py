# src/agent/q_learning_agent.py

from pathlib import Path
from collections import defaultdict
import json
import random

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "trained"

PLOT_DIR = PROJECT_ROOT / "results" / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

ACTIONS = {
    0: "BUY",
    1: "SELL",
    2: "HOLD",
}

NUM_ACTIONS = len(ACTIONS)


# ============================================================
# Q-LEARNING AGENT
# ============================================================

class QLearningAgent:

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        random_seed=42,
    ):

        print("\n" + "=" * 70)
        print("INITIALIZING Q-LEARNING AGENT")
        print("=" * 70)

        # ----------------------------------------------------
        # Learning parameters
        # ----------------------------------------------------

        self.learning_rate = learning_rate

        self.discount_factor = discount_factor

        self.epsilon = epsilon

        self.epsilon_min = epsilon_min

        self.epsilon_decay = epsilon_decay

        # ----------------------------------------------------
        # Random generator
        # ----------------------------------------------------

        self.random_generator = random.Random(random_seed)

        # ----------------------------------------------------
        # Q-Table
        # ----------------------------------------------------

        self.q_table = defaultdict(
            lambda: np.zeros(NUM_ACTIONS, dtype=np.float64)
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        self.total_updates = 0

        self.total_actions = 0

        self.exploration_actions = 0

        self.exploitation_actions = 0

        self.epsilon_history = []

        self.q_value_history = []

        self.reward_history = []

        # ----------------------------------------------------
        # Configuration display
        # ----------------------------------------------------

        print("\nAgent Configuration:")

        print(f"Learning Rate:       {self.learning_rate}")

        print(f"Discount Factor:     {self.discount_factor}")

        print(f"Initial Epsilon:     {self.epsilon}")

        print(f"Minimum Epsilon:     {self.epsilon_min}")

        print(f"Epsilon Decay:       {self.epsilon_decay}")

        print(f"Available Actions:   {ACTIONS}")

        print("\nQ-Learning Agent initialized successfully.")

    # ========================================================
    # STATE VALIDATION
    # ========================================================

    def validate_state(self, state):

        if not isinstance(state, tuple):

            raise TypeError(
                "State must be a tuple."
            )

        if len(state) == 0:

            raise ValueError(
                "State cannot be empty."
            )

        return state

    # ========================================================
    # GET Q-VALUES
    # ========================================================

    def get_q_values(self, state):

        state = self.validate_state(state)

        return self.q_table[state]

    # ========================================================
    # ACTION SELECTION
    # ========================================================

    def choose_action(
        self,
        state,
        valid_actions=None,
        training=True,
    ):

        state = self.validate_state(state)

        self.total_actions += 1

        # ----------------------------------------------------
        # Available actions
        # ----------------------------------------------------

        if valid_actions is None:

            valid_actions = list(ACTIONS.keys())

        if len(valid_actions) == 0:

            raise ValueError(
                "No valid actions available."
            )

        # ----------------------------------------------------
        # Exploration
        # ----------------------------------------------------

        if training and self.random_generator.random() < self.epsilon:

            action = self.random_generator.choice(
                valid_actions
            )

            self.exploration_actions += 1

            return action

        # ----------------------------------------------------
        # Exploitation
        # ----------------------------------------------------

        q_values = self.get_q_values(state)

        valid_q_values = {
            action: q_values[action]
            for action in valid_actions
        }

        best_q_value = max(
            valid_q_values.values()
        )

        best_actions = [
            action
            for action, q_value in valid_q_values.items()
            if q_value == best_q_value
        ]

        action = self.random_generator.choice(
            best_actions
        )

        self.exploitation_actions += 1

        return action

    # ========================================================
    # Q-LEARNING UPDATE
    # ========================================================

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done=False,
        next_valid_actions=None,
    ):

        state = self.validate_state(state)

        next_state = self.validate_state(next_state)

        # ----------------------------------------------------
        # Current Q-Value
        # ----------------------------------------------------

        current_q_value = self.q_table[state][action]

        # ----------------------------------------------------
        # Future Q-Value
        # ----------------------------------------------------

        if done:

            max_future_q_value = 0.0

        else:

            next_q_values = self.q_table[next_state]

            if next_valid_actions is None:

                max_future_q_value = np.max(
                    next_q_values
                )

            elif len(next_valid_actions) == 0:

                max_future_q_value = 0.0

            else:

                max_future_q_value = max(
                    next_q_values[action]
                    for action in next_valid_actions
                )

        # ----------------------------------------------------
        # Q-Learning Formula
        # ----------------------------------------------------

        target = (
            reward
            + self.discount_factor
            * max_future_q_value
        )

        td_error = (
            target
            - current_q_value
        )

        new_q_value = (
            current_q_value
            + self.learning_rate
            * td_error
        )

        # ----------------------------------------------------
        # Update Q-Table
        # ----------------------------------------------------

        self.q_table[state][action] = new_q_value

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        self.total_updates += 1

        self.reward_history.append(
            reward
        )

        self.q_value_history.append(
            new_q_value
        )

        self.epsilon_history.append(
            self.epsilon
        )

        return {
            "current_q_value": float(current_q_value),
            "target": float(target),
            "td_error": float(td_error),
            "new_q_value": float(new_q_value),
        }

    # ========================================================
    # EPSILON DECAY
    # ========================================================

    def decay_epsilon(self):

        previous_epsilon = self.epsilon

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

        print(
            f"Epsilon: "
            f"{previous_epsilon:.5f} "
            f"-> "
            f"{self.epsilon:.5f}"
        )

        return self.epsilon

    # ========================================================
    # RESET TRAINING STATISTICS
    # ========================================================

    def reset_statistics(self):

        print("\nResetting agent statistics...")

        self.total_updates = 0

        self.total_actions = 0

        self.exploration_actions = 0

        self.exploitation_actions = 0

        self.epsilon_history = []

        self.q_value_history = []

        self.reward_history = []

        print("Statistics reset successfully.")

    # ========================================================
    # AGENT SUMMARY
    # ========================================================

    def print_summary(self):

        print("\n" + "=" * 70)
        print("Q-LEARNING AGENT SUMMARY")
        print("=" * 70)

        print("\nAgent Statistics:")

        print(
            f"Unique States Learned: "
            f"{len(self.q_table):,}"
        )

        print(
            f"Total Q-Updates: "
            f"{self.total_updates:,}"
        )

        print(
            f"Total Actions: "
            f"{self.total_actions:,}"
        )

        print(
            f"Exploration Actions: "
            f"{self.exploration_actions:,}"
        )

        print(
            f"Exploitation Actions: "
            f"{self.exploitation_actions:,}"
        )

        print(
            f"Current Epsilon: "
            f"{self.epsilon:.5f}"
        )

        if self.total_actions > 0:

            exploration_percentage = (
                self.exploration_actions
                / self.total_actions
                * 100
            )

            exploitation_percentage = (
                self.exploitation_actions
                / self.total_actions
                * 100
            )

            print(
                f"Exploration: "
                f"{exploration_percentage:.2f}%"
            )

            print(
                f"Exploitation: "
                f"{exploitation_percentage:.2f}%"
            )

    # ========================================================
    # SAVE AGENT
    # ========================================================

    def save(self, filename="q_learning_agent.json"):

        print("\n" + "=" * 70)
        print("SAVING Q-LEARNING AGENT")
        print("=" * 70)

        filepath = MODEL_DIR / filename

        serialized_q_table = {}

        for state, q_values in self.q_table.items():

            state_key = ",".join(
                map(str, state)
            )

            serialized_q_table[state_key] = (
                q_values.tolist()
            )

        model_data = {

            "learning_rate": self.learning_rate,

            "discount_factor": self.discount_factor,

            "epsilon": self.epsilon,

            "epsilon_min": self.epsilon_min,

            "epsilon_decay": self.epsilon_decay,

            "actions": ACTIONS,

            "q_table": serialized_q_table,

            "statistics": {

                "total_updates": self.total_updates,

                "total_actions": self.total_actions,

                "exploration_actions": self.exploration_actions,

                "exploitation_actions": self.exploitation_actions,

            }

        }

        with open(
            filepath,
            "w"
        ) as file:

            json.dump(
                model_data,
                file,
                indent=4
            )

        print(f"Agent saved successfully:")
        print(filepath)

        return filepath

    # ========================================================
    # LOAD AGENT
    # ========================================================

    def load(self, filename="q_learning_agent.json"):

        print("\n" + "=" * 70)
        print("LOADING Q-LEARNING AGENT")
        print("=" * 70)

        filepath = MODEL_DIR / filename

        if not filepath.exists():

            raise FileNotFoundError(
                f"Model not found: {filepath}"
            )

        with open(
            filepath,
            "r"
        ) as file:

            model_data = json.load(file)

        # ----------------------------------------------------
        # Load configuration
        # ----------------------------------------------------

        self.learning_rate = model_data["learning_rate"]

        self.discount_factor = model_data["discount_factor"]

        self.epsilon = model_data["epsilon"]

        self.epsilon_min = model_data["epsilon_min"]

        self.epsilon_decay = model_data["epsilon_decay"]

        # ----------------------------------------------------
        # Load Q-Table
        # ----------------------------------------------------

        self.q_table = defaultdict(
            lambda: np.zeros(NUM_ACTIONS, dtype=np.float64)
        )

        for state_key, q_values in model_data["q_table"].items():

            state = tuple(
                int(value)
                for value in state_key.split(",")
            )

            self.q_table[state] = np.array(
                q_values,
                dtype=np.float64
            )

        # ----------------------------------------------------
        # Load statistics
        # ----------------------------------------------------

        statistics = model_data.get(
            "statistics",
            {}
        )

        self.total_updates = statistics.get(
            "total_updates",
            0
        )

        self.total_actions = statistics.get(
            "total_actions",
            0
        )

        self.exploration_actions = statistics.get(
            "exploration_actions",
            0
        )

        self.exploitation_actions = statistics.get(
            "exploitation_actions",
            0
        )

        print(f"Agent loaded successfully:")
        print(filepath)

        print(
            f"Loaded states: "
            f"{len(self.q_table):,}"
        )

        return self

    # ========================================================
    # VISUALIZATION
    # ========================================================

    def create_visualizations(self):

        print("\n" + "=" * 70)
        print("GENERATING AGENT VISUALIZATIONS")
        print("=" * 70)

        # ----------------------------------------------------
        # Q-Value Updates
        # ----------------------------------------------------

        if len(self.q_value_history) > 0:

            print("\nGenerating Q-Value learning visualization...")

            fig, ax = plt.subplots(
                figsize=(14, 7)
            )

            ax.plot(
                self.q_value_history,
                label="Q-Value",
            )

            ax.set_title(
                "Q-Learning Value Updates"
            )

            ax.set_xlabel(
                "Update Number"
            )

            ax.set_ylabel(
                "Q-Value"
            )

            ax.legend()

            ax.grid(
                True,
                alpha=0.3
            )

            plt.tight_layout()

            output_file = (
                PLOT_DIR
                / "q_learning_value_updates.png"
            )

            plt.savefig(
                output_file,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            print(f"Saved: {output_file}")

        # ----------------------------------------------------
        # Epsilon
        # ----------------------------------------------------

        if len(self.epsilon_history) > 0:

            print("\nGenerating epsilon visualization...")

            fig, ax = plt.subplots(
                figsize=(14, 7)
            )

            ax.plot(
                self.epsilon_history,
                label="Epsilon",
            )

            ax.set_title(
                "Exploration Rate During Learning"
            )

            ax.set_xlabel(
                "Update Number"
            )

            ax.set_ylabel(
                "Epsilon"
            )

            ax.grid(
                True,
                alpha=0.3
            )

            ax.legend()

            plt.tight_layout()

            output_file = (
                PLOT_DIR
                / "q_learning_epsilon.png"
            )

            plt.savefig(
                output_file,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            print(f"Saved: {output_file}")

        # ----------------------------------------------------
        # Reward
        # ----------------------------------------------------

        if len(self.reward_history) > 0:

            print("\nGenerating reward visualization...")

            rewards = np.array(
                self.reward_history
            )

            cumulative_rewards = np.cumsum(
                rewards
            )

            fig, ax = plt.subplots(
                figsize=(14, 7)
            )

            ax.plot(
                cumulative_rewards,
                label="Cumulative Reward",
            )

            ax.set_title(
                "Cumulative Reward During Learning"
            )

            ax.set_xlabel(
                "Update Number"
            )

            ax.set_ylabel(
                "Cumulative Reward"
            )

            ax.grid(
                True,
                alpha=0.3
            )

            ax.legend()

            plt.tight_layout()

            output_file = (
                PLOT_DIR
                / "q_learning_cumulative_reward.png"
            )

            plt.savefig(
                output_file,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            print(f"Saved: {output_file}")

        print("\nVisualization generation completed.")

    # ========================================================
    # Q-TABLE VISUALIZATION
    # ========================================================

    def visualize_q_table(self):

        print("\nGenerating Q-Table visualization...")

        if len(self.q_table) == 0:

            print("Q-Table is empty.")

            return

        states = list(
            self.q_table.keys()
        )

        q_values = np.array(
            [
                self.q_table[state]
                for state in states
            ]
        )

        fig, ax = plt.subplots(
            figsize=(14, 8)
        )

        im = ax.imshow(
            q_values,
            aspect="auto",
        )

        ax.set_title(
            "Q-Table Heatmap"
        )

        ax.set_xlabel(
            "Actions"
        )

        ax.set_ylabel(
            "States"
        )

        ax.set_xticks(
            range(NUM_ACTIONS)
        )

        ax.set_xticklabels(
            [
                ACTIONS[action]
                for action in range(NUM_ACTIONS)
            ]
        )

        plt.colorbar(
            im,
            ax=ax,
            label="Q-Value"
        )

        plt.tight_layout()

        output_file = (
            PLOT_DIR
            / "q_table_heatmap.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved: {output_file}")


# ============================================================
# STANDALONE TEST
# ============================================================

def test_q_learning_agent():

    print("\n")
    print("=" * 70)
    print("Q-LEARNING AGENT TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Initialize Agent
    # --------------------------------------------------------

    print("\nTEST 1: Agent Initialization")

    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        random_seed=42,
    )

    # --------------------------------------------------------
    # Test State
    # --------------------------------------------------------

    state = (
        2,
        3,
        1,
        4,
        2,
        0,
    )

    next_state = (
        3,
        3,
        2,
        4,
        1,
        1,
    )

    print("\nTEST 2: Initial Q-Values")

    print(
        f"State: {state}"
    )

    print(
        f"Q-Values: {agent.get_q_values(state)}"
    )

    # --------------------------------------------------------
    # Test Exploration
    # --------------------------------------------------------

    print("\nTEST 3: Exploration")

    for i in range(10):

        action = agent.choose_action(
            state,
            training=True
        )

        print(
            f"Action {i + 1}: "
            f"{ACTIONS[action]}"
        )

    # --------------------------------------------------------
    # Test Exploitation
    # --------------------------------------------------------

    print("\nTEST 4: Exploitation")

    agent.epsilon = 0.0

    agent.q_table[state] = np.array(
        [
            0.10,
            0.20,
            0.80,
        ]
    )

    action = agent.choose_action(
        state,
        training=True
    )

    print(
        f"Q-Values: "
        f"{agent.q_table[state]}"
    )

    print(
        f"Selected Action: "
        f"{ACTIONS[action]}"
    )

    assert action == 2

    print(
        "Exploitation test passed."
    )

    # --------------------------------------------------------
    # Test Q-Update
    # --------------------------------------------------------

    print("\nTEST 5: Q-Learning Update")

    agent.epsilon = 0.0

    state = (
        2,
        3,
        1,
        4,
        2,
        0,
    )

    next_state = (
        3,
        3,
        2,
        4,
        1,
        1,
    )

    agent.q_table[state] = np.array(
        [
            0.50,
            0.20,
            0.10,
        ]
    )

    agent.q_table[next_state] = np.array(
        [
            0.30,
            0.40,
            0.80,
        ]
    )

    reward = 0.10

    result = agent.update(
        state=state,
        action=0,
        reward=reward,
        next_state=next_state,
        done=False,
    )

    print("\nUpdate Results:")

    print(
        f"Current Q-Value: "
        f"{result['current_q_value']:.4f}"
    )

    print(
        f"Target: "
        f"{result['target']:.4f}"
    )

    print(
        f"TD Error: "
        f"{result['td_error']:.4f}"
    )

    print(
        f"New Q-Value: "
        f"{result['new_q_value']:.4f}"
    )

    # --------------------------------------------------------
    # Test Epsilon Decay
    # --------------------------------------------------------

    print("\nTEST 6: Epsilon Decay")

    agent.epsilon = 1.0

    for episode in range(10):

        agent.decay_epsilon()

    print(
        f"Final Epsilon: "
        f"{agent.epsilon:.5f}"
    )

    # --------------------------------------------------------
    # Simulated Learning
    # --------------------------------------------------------

    print("\nTEST 7: Simulated Learning")

    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        random_seed=42,
    )

    states = [
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1, 0),
        (2, 2, 2, 2, 2, 0),
        (3, 3, 3, 3, 3, 1),
        (4, 4, 4, 4, 4, 1),
    ]

    print(
        "\nSimulating 500 learning updates..."
    )

    for episode in range(500):

        state = random.choice(
            states
        )

        next_state = random.choice(
            states
        )

        action = agent.choose_action(
            state,
            training=True
        )

        reward = random.uniform(
            -0.02,
            0.02
        )

        agent.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=False,
        )

        agent.decay_epsilon()

    print(
        "Simulated learning completed."
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nTEST 8: Agent Summary")

    agent.print_summary()

    # --------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------

    print("\nTEST 9: Visualizations")

    agent.create_visualizations()

    agent.visualize_q_table()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\nTEST 10: Save Agent")

    agent.save(
        "q_learning_agent_test.json"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\nTEST 11: Load Agent")

    loaded_agent = QLearningAgent(
        random_seed=42
    )

    loaded_agent.load(
        "q_learning_agent_test.json"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "ALL Q-LEARNING AGENT TESTS COMPLETED"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    test_q_learning_agent()