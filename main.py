import optuna
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import dump

from services.loader import DataLoader
from services.processor import DataProcessor
from services.optimizer import LightGBMOptimizer
from services.scorer import LightGBMScorer
from config import *

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

    optimizer = LightGBMOptimizer(data_processor)
    study = optuna.create_study(direction=OPTIMIZATION_DIRECTION)
    study.optimize(lambda trial: optimizer.objective(trial, BASE_PARAMS), n_trials=N_TRIALS)

    best_params = study.best_trial.params
    best_num_boost_round = study.best_trial.user_attrs["num_boost_round"]

    print('best_num_boost_round:', best_num_boost_round)
    print('Best trial:', best_params)

    scorer = LightGBMScorer(data_processor.X_test, data_processor.y_test, data_processor.class_label_dict)

    # Train and score models
    default_lgbm, evals_result = optimizer.train_model(BASE_PARAMS)
    _ = scorer.plot_results(default_lgbm, evals_result, "default")

    hp_params = best_params.copy()
    hp_params.update(BASE_PARAMS)
    hp_params["num_boost_round"] = best_num_boost_round
    optimized_lgbm, evals_result = optimizer.train_model(hp_params)
    _ = scorer.plot_results(optimized_lgbm, evals_result, "optimized")

    # Score models
    default_lgbm_df = scorer.score_model(default_lgbm, "default_lgbm", data_processor.label_encoder)
    optimized_lgbm_df = scorer.score_model(optimized_lgbm, "optimized_lgbm", data_processor.label_encoder)

    scores_df = (
        pd.concat([default_lgbm_df, optimized_lgbm_df])
        .reset_index(drop=True)
        .rename(columns={"index": "label"})
    )

    # Dump models and scores_df
    dump(default_lgbm, 'dump/default_lgbm.joblib')
    dump(optimized_lgbm, 'dump/optimized_lgbm.joblib')
    scores_df.to_csv('dump/scores_df.csv', index=False)

    print("Models and scores dataframe have been saved to the 'dump' directory.")

if __name__ == "__main__":
    main()


# loader = DataLoader(DATA_PATH)
# label_encoder = LabelEncoder()
# data_processor = DataProcessor(id_column=IDENTIFIER_COLUMN,
#                  target_columns=RAW_TARGET_COLUMNS,
#                  loader= loader, 
#                  label_encoder=label_encoder,
#                  random_state=RANDOM_STATE,
#                  test_size= TEST_SIZE,
#                  shuffle = SHUFFLE_DATA,
#                  )
# data_processor.run()
# 
# 
# optimizer = LightGBMOptimizer(data_processor)
# study = optuna.create_study(direction=OPTIMIZATION_DIRECTION)
# study.optimize(lambda trial: optimizer.objective(trial, BASE_PARAMS), n_trials=N_TRIALS)
#  
# best_params = study.best_trial.params
# best_num_boost_round = study.best_trial.user_attrs["num_boost_round"]
# 
# print('best_num_boost_round:', best_num_boost_round)
# print('Best trial:', best_params)
# 
# 
# scorer = LightGBMScorer(data_processor.X_test, data_processor.y_test, data_processor.class_label_dict)
# 
# # benchmark LightGBM
# default_lgbm, evals_result = optimizer.train_model(BASE_PARAMS)
# default_report = scorer.plot_results(default_lgbm, evals_result, "default")
# 
# # optimized LightGBM
# hp_params = best_params
# hp_params.update(BASE_PARAMS)
# hp_params.update({"num_boost_round": best_num_boost_round})
# optimized_lgbm, evals_result = optimizer.train_model(hp_params)
# optimized_report = scorer.plot_results(optimized_lgbm, evals_result, "optimized")
# 
# # score models
# default_lgbm_df = scorer.score_model(default_lgbm, "default_lgbm", data_processor.label_encoder)
# lgbm_df = scorer.score_model(optimized_lgbm, "optimized lgbm", data_processor.label_encoder)
# 
# scores_df = (
#     pd.concat([default_lgbm_df, lgbm_df])
#     .reset_index().rename({"index": "label"}, axis=1)
# )