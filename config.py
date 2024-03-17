import lightgbm
import numpy as np
from typing import Tuple

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
    
# common parameters
RANDOM_STATE = 42
TEST_SIZE = 0.1
SHUFFLE_DATA = True
DATA_PATH = "data"
IDENTIFIER_COLUMN = "id"
TARGET_COLUMN = "target"
RAW_TARGET_COLUMNS = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]

# lightgbm parameters
N_TRIALS = 3
RANDOM_STATE = 42
OPTIMIZATION_DIRECTION = "maximize"

def lgb_auc_avg(y_pred_binarized: np.ndarray, data: lightgbm.Dataset) -> Tuple[str, float, bool]:
    is_higher_better = True
    
    y_true = data.get_label()
    lb = LabelBinarizer()
    y_true_binarized = lb.fit_transform(y_true)

    # Calculate AUC for each class and take the average
    auc_scores = [roc_auc_score(y_true_binarized[:, i], y_pred_binarized[:, i]) for i in range(y_true_binarized.shape[1])]
    average_auc = np.mean(auc_scores)
    return ("my_auc_avg", average_auc, is_higher_better)

BASE_PARAMS = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 7,
    'verbosity': -1,
    'boosting_type': 'gbdt',
    "nfold": 5,
    "stratified": True,
    "shuffle": True,
    "seed": RANDOM_STATE,
    "num_boost_round": 400,
    "callbacks": [lightgbm.early_stopping(40), lightgbm.log_evaluation(40)],
    "eval_train_metric": True,
    "return_cvbooster":True,
    "feature_pre_filter": False,
    "feval":[lgb_auc_avg],
    # "eval_class_weight": None, #[class_weight_dict],
}