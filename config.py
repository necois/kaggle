# Import standard libraries
from typing import Tuple, List, Any

# Import third-party libraries
import numpy as np
import lightgbm
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score

# Common parameters
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.1
SHUFFLE_DATA: bool = True
DATA_PATH: str = "data"
IDENTIFIER_COLUMN: str = "id"
TARGET_COLUMN: str = "target"
RAW_TARGET_COLUMNS: List[str] = [
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]

# LightGBM parameters
N_TRIALS: int = 3
OPTIMIZATION_DIRECTION: str = "maximize"
LGBM_METRIC_NAME: str = "my_auc_avg"


def lgb_auc_avg(
    y_pred_binarized: np.ndarray, data: lightgbm.Dataset
) -> Tuple[str, float, bool]:
    """Calculate the average AUC for a multiclass problem in LightGBM.

    Args:
        y_pred_binarized (np.ndarray): The binarized predictions from the model.
        data (lightgbm.Dataset): The LightGBM dataset object containing the true labels.

    Returns:
        Tuple[str, float, bool]: A tuple containing the custom metric name, the calculated average AUC, and a boolean indicating if higher is better.
    """
    is_higher_better: bool = True
    y_true = data.get_label()
    lb: LabelBinarizer = LabelBinarizer()
    y_true_binarized: np.ndarray = lb.fit_transform(y_true)

    # Calculate AUC for each class and take the average
    auc_scores: List = [
        roc_auc_score(y_true_binarized[:, i], y_pred_binarized[:, i])
        for i in range(y_true_binarized.shape[1])
    ]
    average_auc: float = np.mean(auc_scores)
    return (LGBM_METRIC_NAME, average_auc, is_higher_better)


BASE_PARAMS: dict = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_class": 7,
    "verbosity": -1,
    "boosting_type": "gbdt",
    "nfold": 5,
    "stratified": True,
    "shuffle": True,
    "seed": RANDOM_STATE,
    "num_boost_round": 400,
    "callbacks": [lightgbm.early_stopping(40), lightgbm.log_evaluation(40)],
    "eval_train_metric": True,
    "return_cvbooster": True,
    "feature_pre_filter": False,
    "feval": [lgb_auc_avg],
    # "eval_class_weight": None, # Uncomment and set your class weights if needed
}
