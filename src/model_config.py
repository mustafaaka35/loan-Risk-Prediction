# ---
# jupyter:
#   jupytext:
#     formats: ipynb,src//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7222, "status": "ok", "timestamp": 1786304154633, "user": {"displayName": "Mustafa Aka", "userId": "16928966536968889021"}, "user_tz": -180} id="vwHdH9yK5VMq" outputId="7379a40f-327f-4ed1-921b-ad73e7a0c0eb"
# !pip install jupytext --quiet
import os
drive_path="/content/drive/MyDrive/loan_my_ml_project"
IPYNB_path=f"{drive_path}/notebooks/model_config.ipynb"

# !jupytext --set-formats ipynb,src//py:percent {IPYNB_path}

print("otomatık eslenme saglandı.")


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1786304154651, "user": {"displayName": "Mustafa Aka", "userId": "16928966536968889021"}, "user_tz": -180} id="DlI-ZhKke57S" outputId="4bfbb2e1-483d-4ae9-8016-acf0909c6790"
# %%writefile {project_path}/src/model_config.py"
class ModelConfig:

    """
    Configuration class for machine learning models.

    This class stores all project settings in one place.
    If a parameter needs to be changed, it is modified here only.
    """

    # ==========================================================
    # DATASET SETTINGS
    # ==========================================================

    # Target column (label)
    TARGET = "loan_status"

    # Percentage of test data
    TEST_SIZE = 0.20

    # Random seed for reproducibility
    RANDOM_STATE = 42

    # Keep class distribution the same in train and test sets
    STRATIFY = True


    # ==========================================================
    # SCALING SETTINGS
    # ==========================================================

    # Apply feature scaling before training
    USE_SCALER = True


    # ==========================================================
    # LOGISTIC REGRESSION PARAMETERS
    # ==========================================================

    LOGISTIC_PARAMS = {

        # Regularization strength
        "C": 1.0,

        # L2 Regularization
        "penalty": "l2",

        # Optimization algorithm
        "solver": "lbfgs",

        # Maximum training iterations
        "max_iter": 1000,

        #Random seed
        "random_state": RANDOM_STATE,

        #class weight
        "class_weight":"balanced"

        }


    # ==========================================================
    # RANDOM FOREST PARAMETERS
    # ==========================================================
    RANDOM_FOREST_PARAMS = {
        # Number of trees"
        "n_estimators": 100,

        # Split criterion
        "criterion": "entropy",

        # Maximum tree depth
        "max_depth": 9,

        # Minimum samples required to split
        "min_samples_split": 2,

        # Minimum samples in a leaf
        "min_samples_leaf": 1,

        # Handle class balance
        "class_weight": {0:5,1:1},

        # Random seed
        "random_state": RANDOM_STATE,

        # Use all CPU cores
        "n_jobs": -1,

        # Bootstrap sampling
        "bootstrap": True
        }


    # ==========================================================
    # PROJECT PATHS
    # ==========================================================

    MODEL_PATH = "/content/drive/MyDrive/loan_my_ml_project/models/"

    LOG_PATH = "/content/drive/MyDrive/loan_my_ml_project/logs/"

    FIGURE_PATH = "/content/drive/MyDrive/loan_my_ml_project/figures/"

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2827, "status": "ok", "timestamp": 1786304166578, "user": {"displayName": "Mustafa Aka", "userId": "16928966536968889021"}, "user_tz": -180} id="abkE6QpV-WVG" outputId="584c820a-e023-4bfd-fcf3-c6087f5ddc45"
# !jupytext --to py:percent --output /content/drive/MyDrive/loan_my_ml_project/src/model_config.py {IPYNB_path}
