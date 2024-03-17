import lightgbm
import pandas as pd

from typing import Tuple, Dict
from services.processor import DataProcessor 

class LightGBMOptimizer:
    def __init__(self, data_processor: DataProcessor):
        self.label_encoder =  data_processor.label_encoder
        self.data_train = lightgbm.Dataset(data_processor.X_train, data_processor.y_train, free_raw_data=False)
        self.data_test = lightgbm.Dataset(data_processor.X_test, data_processor.y_test, free_raw_data=False, reference=self.data_train)

    def train_model(self, params: Dict) -> Tuple[lightgbm.Booster, Dict]:
        evals_result: Dict = {}
        booster = lightgbm.train(params,
                  train_set=self.data_train,
                  valid_sets=[self.data_train, self.data_test],
                  valid_names=["train", "set"],
                  num_boost_round=params["num_boost_round"],
                  callbacks=params["callbacks"]+[lightgbm.record_evaluation(evals_result)],
                  feval=params["feval"],
                  # eval_class_weight=params["class_weight_dict"],
                 )
        return booster, evals_result

    # def fit_and_plot(self,
    #     params: dict, 
    #     X_test: pd.DataFrame, 
    #     y_test: pd.Series,
    #     labels: List[str])-> Tuple[lightgbm.Booster, str]:
    #     evals_result: dict = {}
    #     booster = lightgbm.train(params,
    #               train_set=self.data_train,
    #               valid_sets=[self.data_train, self.data_test],
    #               valid_names=["train", "set"],
    #               num_boost_round=params["num_boost_round"],
    #               callbacks=params["callbacks"]+[lightgbm.record_evaluation(evals_result)],
    #               feval=params["feval"],
    #               # eval_class_weight=params["class_weight_dict"],
    #              )

    #     y_test_pred = booster.predict(X_test, raw_score=False).argmax(axis=1)

    #     lightgbm.plot_importance(booster, figsize=(12, 6), importance_type="gain")
    #     lightgbm.plot_metric(evals_result, "my_auc_avg")


    #     cnf_matrix = confusion_matrix(y_test, y_test_pred, labels=labels, normalize="true")
    #     cnf_disp = ConfusionMatrixDisplay(cnf_matrix, display_labels=labels)

    #     cnf_disp.plot()
    #     plt.xticks(rotation=90)

    #     report = classification_report(y_test, y_test_pred)
    #     return booster, report


    def objective(self, trial, base_params, booster=None):
        params = {
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 128),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1.0, log=True),
        }

        params.update(base_params)

        results = lightgbm.cv(params,
                              train_set=self.data_train,
                              nfold=base_params["nfold"],
                              stratified=base_params["stratified"],
                              shuffle=base_params["shuffle"],
                              seed=base_params["seed"],
                              num_boost_round=base_params["num_boost_round"],
                              callbacks=base_params["callbacks"],
                              return_cvbooster=base_params["return_cvbooster"],
                              feval=base_params["feval"],
                              init_model=booster,
                              # eval_class_weight=base_params["class_weight_dict"],
                             )

        num_boost_round = pd.Series(results["valid my_auc_avg-mean"]).idxmax()

        trial.set_user_attr("num_boost_round", num_boost_round)
        trial.set_user_attr("cvbooster", results["cvbooster"])

        return max(results["valid my_auc_avg-mean"])

