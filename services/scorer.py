import lightgbm
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import dump
from typing import Dict, List, Any
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

class LightGBMScorer:
    DUMP_DIR: Path = Path("dump")

    def __init__(self, X_test: pd.DataFrame, y_test: pd.Series, class_label_dict: Dict[int, str]) -> None:
        self.DUMP_DIR.mkdir(parents=True, exist_ok=True)

        self.X_test: pd.DataFrame = X_test
        self.y_test: pd.Series = y_test
        self.class_label_dict: Dict[int, str] = class_label_dict

        self.classes: List[int] = list(self.class_label_dict.keys())
        self.labels: List[str] = list(self.class_label_dict.values())

    def plot_results(
        self,
        booster: lightgbm.Booster,
        evals_result: Dict[str, Any],
        metric_name: str,
        model_name: str,
    ) -> str:

        y_test_pred = booster.predict(self.X_test, raw_score=False).argmax(axis=1)

        # feature importance
        lightgbm.plot_importance(booster, figsize=(12, 6), importance_type="gain")
        plt.savefig(self.DUMP_DIR / f"{model_name}_feature_importance.png")
        plt.close()

        # metric plot
        lightgbm.plot_metric(evals_result, metric=metric_name)
        plt.savefig(self.DUMP_DIR / f"{model_name}_metric_plot.png")
        plt.close()

        # confusion matrix
        cnf_matrix = confusion_matrix(
            self.y_test, y_test_pred, labels=self.classes, normalize="true"
        )
        cnf_disp = ConfusionMatrixDisplay(cnf_matrix, display_labels=self.labels)
        cnf_disp.plot()
        plt.xticks(rotation=90)
        plt.savefig(self.DUMP_DIR / f"{model_name}_confusion_matrix.png")
        plt.close()

        report = classification_report(self.y_test, y_test_pred)
        return report

    def save_and_score_model(
        self, 
        booster: lightgbm.Booster, 
        model_name: str, 
        label_encoder: LabelEncoder, 
    ) -> pd.DataFrame:

        dump(booster, self.DUMP_DIR / f"{model_name}.joblib")

        y_test_binarized = label_binarize(self.y_test, classes=self.classes)

        roc: Dict[str, float] = {}
        labels = label_encoder.inverse_transform(self.classes)
        for idx, label in enumerate(labels):
            y_score = booster.predict(self.X_test)[:, idx]
            roc[label] = roc_auc_score(
                y_test_binarized[:, idx], y_score, multi_class="ovo"
            )
        scores_df = pd.DataFrame.from_dict(roc, orient="index", columns=["auc"]).assign(
            model_name=model_name
        )
        scores_df.to_csv(self.DUMP_DIR / f"{model_name}_scores_df.csv", index=False)

        return scores_df
