"""
Trade RL - Complete Project Execution Script

Runs the complete Reinforcement Learning trading pipeline:

1. Download Market Data
2. Data Preprocessing
3. Feature Engineering
4. Data Splitting
5. Trading Environment Testing
6. Q-Learning Agent Testing
7. TD3 Agent Testing
8. TD3 Training
9. TD3 Evaluation
10. TD3 Diagnostics

Execution:
    python run_project.py
"""

import subprocess
import sys
import time

from pathlib import Path


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

PYTHON_EXECUTABLE = sys.executable


# ============================================================
# EXECUTION STEPS
# ============================================================

PROJECT_STEPS = [

    {
        "name": "DOWNLOAD MARKET DATA",
        "command": [
            PYTHON_EXECUTABLE,
            "data/download_data.py",
        ],
    },

    {
        "name": "DATA PREPROCESSING",
        "command": [
            PYTHON_EXECUTABLE,
            "-m",
            "src.data.data_preprocessor",
        ],
    },

    {
        "name": "FEATURE ENGINEERING",
        "command": [
            PYTHON_EXECUTABLE,
            "-m",
            "src.features.feature_engineering",
        ],
    },

    {
        "name": "DATA SPLITTING",
        "command": [
            PYTHON_EXECUTABLE,
            "-m",
            "src.data.data_splitter",
        ],
    },

    {
        "name": "TRADING ENVIRONMENT TEST",
        "command": [
            PYTHON_EXECUTABLE,
            "main.py",
        ],
    },

    {
        "name": "Q-LEARNING AGENT TEST",
        "command": [
            PYTHON_EXECUTABLE,
            "-m",
            "src.agent.q_learning_agent",
        ],
    },

    {
        "name": "TD3 AGENT TEST",
        "command": [
            PYTHON_EXECUTABLE,
            "-m",
            "src.agent.td3_agent",
        ],
    },

    {
        "name": "TD3 MODEL TRAINING",
        "command": [
            PYTHON_EXECUTABLE,
            "-m",
            "src.training.td3_trainer",
        ],
    },

    {
        "name": "TD3 MODEL EVALUATION",
        "command": [
            PYTHON_EXECUTABLE,
            "-m",
            "src.evaluation.td3_evaluator",
        ],
    },

    {
        "name": "TD3 DIAGNOSTIC ANALYSIS",
        "command": [
            PYTHON_EXECUTABLE,
            "-m",
            "src.evaluation.td3_diagnostics",
        ],
    },
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_header(message):

    print("\n")

    print("=" * 100)

    print(message.center(100))

    print("=" * 100)


# ============================================================
# RUN INDIVIDUAL STEP
# ============================================================

def run_step(step_number, total_steps, step):

    step_name = step["name"]

    command = step["command"]

    print_header(
        f"STEP {step_number}/{total_steps}: {step_name}"
    )

    print("\nCommand:")

    print(
        " ".join(command)
    )

    print("\nStarting execution...")

    start_time = time.time()

    try:

        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
        )

        execution_time = (
            time.time() - start_time
        )

        print("\n")

        print("-" * 100)

        print(
            f"STEP COMPLETED SUCCESSFULLY: {step_name}"
        )

        print(
            f"Execution Time: {execution_time:.2f} seconds"
        )

        print("-" * 100)

        return True

    except subprocess.CalledProcessError as error:

        execution_time = (
            time.time() - start_time
        )

        print("\n")

        print("=" * 100)

        print(
            f"STEP FAILED: {step_name}"
        )

        print(
            f"Exit Code: {error.returncode}"
        )

        print(
            f"Execution Time: {execution_time:.2f} seconds"
        )

        print("=" * 100)

        return False

    except Exception as error:

        print("\n")

        print("=" * 100)

        print(
            f"UNEXPECTED ERROR: {step_name}"
        )

        print(
            f"Error: {error}"
        )

        print("=" * 100)

        return False


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print_header(
        "TRADE RL - COMPLETE PROJECT EXECUTION"
    )

    print(
        f"\nProject Directory: {PROJECT_ROOT}"
    )

    print(
        f"Python Executable: {PYTHON_EXECUTABLE}"
    )

    print(
        f"Total Execution Steps: {len(PROJECT_STEPS)}"
    )

    print("\n")

    print("=" * 100)

    print(
        "THE COMPLETE TRADE RL PIPELINE WILL NOW EXECUTE"
    )

    print("=" * 100)

    print("\nExecution Sequence:")

    for index, step in enumerate(
        PROJECT_STEPS,
        start=1,
    ):

        print(
            f"{index}. {step['name']}"
        )

    print("\n")

    input(
        "Press ENTER to start the complete project execution..."
    )

    project_start_time = time.time()

    completed_steps = []

    failed_steps = []

    total_steps = len(
        PROJECT_STEPS
    )

    # ========================================================
    # EXECUTE ALL STEPS
    # ========================================================

    for index, step in enumerate(
        PROJECT_STEPS,
        start=1,
    ):

        success = run_step(
            index,
            total_steps,
            step,
        )

        if success:

            completed_steps.append(
                step["name"]
            )

        else:

            failed_steps.append(
                step["name"]
            )

            print_header(
                "PROJECT EXECUTION STOPPED"
            )

            print(
                "\nA step failed."
            )

            print(
                "The remaining steps will not execute."
            )

            print(
                "\nPlease fix the error and run the script again."
            )

            break

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_execution_time = (
        time.time() - project_start_time
    )

    print_header(
        "TRADE RL EXECUTION SUMMARY"
    )

    print(
        f"\nCompleted Steps: {len(completed_steps)}"
    )

    print(
        f"Failed Steps: {len(failed_steps)}"
    )

    print(
        f"Total Execution Time: "
        f"{total_execution_time / 60:.2f} minutes"
    )

    # --------------------------------------------------------
    # SUCCESSFUL STEPS
    # --------------------------------------------------------

    if completed_steps:

        print("\n")

        print("-" * 100)

        print(
            "SUCCESSFUL STEPS"
        )

        print("-" * 100)

        for step in completed_steps:

            print(
                f"[SUCCESS] {step}"
            )

    # --------------------------------------------------------
    # FAILED STEPS
    # --------------------------------------------------------

    if failed_steps:

        print("\n")

        print("-" * 100)

        print(
            "FAILED STEPS"
        )

        print("-" * 100)

        for step in failed_steps:

            print(
                f"[FAILED] {step}"
            )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    if not failed_steps:

        print_header(
            "COMPLETE PROJECT EXECUTION SUCCESSFUL"
        )

        print(
            "\nAll Trade RL pipeline steps completed successfully."
        )

        print(
            "\nGenerated outputs include:"
        )

        print(
            "- Processed market data"
        )

        print(
            "- Engineered features"
        )

        print(
            "- Train / Validation / Test datasets"
        )

        print(
            "- TD3 trained model"
        )

        print(
            "- Training visualizations"
        )

        print(
            "- Validation evaluation results"
        )

        print(
            "- Buy & Hold comparison"
        )

        print(
            "- TD3 diagnostic analysis"
        )

        print("\n")

        print("=" * 100)

        print(
            "TRADE RL PROJECT COMPLETED SUCCESSFULLY"
        )

        print("=" * 100)

    else:

        print_header(
            "TRADE RL EXECUTION FAILED"
        )

        print(
            "\nPlease review the error from the failed step."
        )

        print(
            "Fix the issue and rerun:"
        )

        print(
            "python run_project.py"
        )


# ============================================================
# EXECUTION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()