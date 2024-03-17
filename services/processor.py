import numpy as np
import pandas as pd

from typing import List, Tuple

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from services.loader import DataLoader


class DataProcessor:
    DEFAULT_TARGET_NAME: str = "target"
    def __init__(self, 
                 id_column: str,
                 target_columns:List[str],
                 loader: DataLoader, 
                 label_encoder: LabelEncoder,
                 random_state: int,
                 test_size: float,
                 shuffle: bool,
                 ):
        self.id_column = id_column
        self.raw_target_columns = target_columns
        self.random_state = random_state
        self.test_size = test_size
        self.shuffle = shuffle

        self.loader = loader
        self.label_encoder = label_encoder

        self.class_label_dict: dict = {}
        self.class_weight_dict: dict = {}

        self.labels: List[str] = []
        self.features: List[str] = []
        self.raw_features: List[str] = []
        self.sample_weights: List[float] = []

        self.train_df: pd.DataFrame = pd.DataFrame()
        self.test_df: pd.DataFrame = pd.DataFrame()
        self.X_train: pd.DataFrame = pd.DataFrame()
        self.X_test: pd.DataFrame = pd.DataFrame()

        self.y_train: pd.Series = pd.Series()
        self.y_test: pd.Series = pd.Series()
        self.y_label: pd.Series = pd.Series()

    def compute_features(self, df: pd.DataFrame) -> List[str]:
        return list(set(df.drop(self.id_column, axis=1).columns)-set(self.raw_target_columns))

    @staticmethod
    def compute_symbolic_regression_features(df: pd.DataFrame, raw_features:List[str]) -> pd.DataFrame:
        X1 = df.get("SigmoidOfAreas") #raw_features[1])
        X3 = df.get("Outside_X_Index") #raw_features[3])
        X14 = df.get("Edges_Y_Index") #raw_features[14])
        return df.assign(feature1= np.sqrt((np.sqrt(X3-np.log(X1))+X3)-np.log(X1)),
                         feature2= np.sqrt(np.sqrt(X14)-np.sqrt(X3)))

    def process_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        ids_to_remove = df.loc[lambda df: df.loc[:, self.raw_target_columns].sum(axis=1)>1].get(self.id_column).values

        print("Removed rows", len(ids_to_remove), 100*len(ids_to_remove)/df.shape[0])

        df = df.loc[lambda df: ~df.get(self.id_column).isin(ids_to_remove)].assign(target=None)

        dfs = []
        for raw_target in self.raw_target_columns:
            dfs.append(
                df.loc[lambda df: df.get(raw_target)==1]
                .assign(**{self.DEFAULT_TARGET_NAME:raw_target})
            )
        return pd.concat(dfs).sort_index().drop(self.raw_target_columns,axis=1)
    
    @staticmethod
    def compute_classs_and_sample_weights(y: List) -> Tuple[dict, np.ndarray]:
        classes = np.unique(y)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        sample_weights = np.array([class_weight_dict[class_label] for class_label in y])
        return class_weight_dict, sample_weights


    def run(self):
        train_df = self.loader.train_df 
        test_df = self.loader.test_df 

        self.raw_features = self.compute_features(train_df)

        # hypothesis1: convert to a single multi-class classification task
        self.train_df = self.compute_symbolic_regression_features(train_df, self.raw_features)
        self.test_df = self.compute_symbolic_regression_features(test_df, self.raw_features)

        self.features = self.compute_features(train_df)

        train_df = self.process_targets(train_df)


        # hypothesis2: labels are not ordered so a simple LabelEncoder can be used
        self.y_label = self.label_encoder.fit_transform(train_df.loc[:, [self.DEFAULT_TARGET_NAME]])
        self.class_weight_dict, self.sample_weights = self.compute_classs_and_sample_weights(self.y_label)

        self.labels = self.label_encoder.inverse_transform(list(self.class_weight_dict.keys()))
        self.class_label_dict = dict(zip(self.class_weight_dict.keys(), self.labels))


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(train_df.loc[:, self.features], 
                                                            self.y_label, 
                                                            test_size=self.test_size, 
                                                            shuffle=self.shuffle,
                                                            random_state=self.random_state)