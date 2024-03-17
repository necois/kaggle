import pandas as pd

from pathlib import Path

class DataLoader:
    TEST_FILE_NAME:str = "test.csv"
    TRAIN_FILE_NAME:str = "train.csv"

    def __init__(self, data_path: str):
        self.path_dir = Path.cwd() / data_path
        self.test_df = self.load_data(self.path_dir/self.TEST_FILE_NAME)
        self.train_df = self.load_data(self.path_dir/self.TRAIN_FILE_NAME)

    @staticmethod
    def load_data(path_dir:Path) -> pd.DataFrame:
        return pd.read_csv(path_dir)