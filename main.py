import optuna
import pandas as pd
from sklearn.preprocessing import LabelEncoder


from services.loader import DataLoader
from services.processor import DataProcessor
from services.optimizer import LightGBMOptimizer
from services.scorer import LightGBMScorer
from config import *


DEFAULT_LGBM_NAME: str = "default_lgbm"
OPTIMIZED_LGBM_NAME: str = "optimized_lgbm"


def main():
    loader = DataLoader(DATA_PATH)
    label_encoder = LabelEncoder()
    data_processor = DataProcessor(
        id_column=IDENTIFIER_COLUMN,
        target_columns=RAW_TARGET_COLUMNS,
        loader=loader,
        label_encoder=label_encoder,
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE,
        shuffle=SHUFFLE_DATA,
    )
    data_processor.run()

    # HPs optimization
    optimizer = LightGBMOptimizer(data_processor)
    study = optuna.create_study(direction=OPTIMIZATION_DIRECTION)
    study.optimize(
        lambda trial: optimizer.objective(trial, BASE_PARAMS), n_trials=N_TRIALS
    )

    best_params = study.best_trial.params
    best_num_boost_round = study.best_trial.user_attrs["num_boost_round"]

    print("best_num_boost_round:", best_num_boost_round)
    print("Best trial:", best_params)

    scorer = LightGBMScorer(
        data_processor.X_test, data_processor.y_test, data_processor.class_label_dict
    )

    # Train and score models
    default_lgbm, evals_result = optimizer.train_model(BASE_PARAMS)
    _ = scorer.plot_results(
        default_lgbm, evals_result, LGBM_METRIC_NAME, DEFAULT_LGBM_NAME
    )

    hp_params = best_params.copy()
    hp_params.update(BASE_PARAMS)
    hp_params["num_boost_round"] = best_num_boost_round
    optimized_lgbm, evals_result = optimizer.train_model(hp_params)
    _ = scorer.plot_results(
        optimized_lgbm, evals_result, LGBM_METRIC_NAME, OPTIMIZED_LGBM_NAME
    )

    # Save & score models
    _ = scorer.save_and_score_model(
        default_lgbm,
        DEFAULT_LGBM_NAME,
        data_processor.label_encoder,
    )
    _ = scorer.save_and_score_model(
        optimized_lgbm,
        OPTIMIZED_LGBM_NAME,
        data_processor.label_encoder,
    )


if __name__ == "__main__":
    main()
