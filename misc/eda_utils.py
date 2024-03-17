import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import ydata_profiling as pp
import matplotlib.pyplot as plt

from typing import List

DEFAULT_CORRELATION_METHOD: str = "pearson"
DEFAULT_CORRELATION_THRESHOLD: float = 0.8
DEFAULT_REPORT_FORMAT: str = "html"

ALLOWED_CORRELATION_METHODS: List[str] = ["pearson", "kendall", "spearman"]

def generate_report(df: pd.DataFrame, report_name: str) -> pp.ProfileReport:
    profile = pp.ProfileReport(df, title=report_name, explorative=True)
    profile.to_file(".".join([report_name, DEFAULT_REPORT_FORMAT]))
    return profile

def correlation_matrix(
    df: pd.DataFrame, 
    method: str=DEFAULT_CORRELATION_METHOD, 
    threshold: float=DEFAULT_CORRELATION_THRESHOLD) -> pd.Series:

    # Ensure the method is one of the accepted correlation methods
    if method not in ALLOWED_CORRELATION_METHODS:
        raise ValueError(f"method must be one of the following: {', '.join(ALLOWED_CORRELATION_METHODS)}")
    
    correlation_matrix = df.corr(method=method)
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    _, _ = plt.subplots(figsize=(11, 9))
    
    sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', vmax=1., center=0,
                linewidths=.5, cbar_kws={"shrink": .5}, annot=False, square=False)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.show()

    filtered_correlation = correlation_matrix.where((abs(correlation_matrix) > threshold) & (correlation_matrix != 1.0))
    pairs = filtered_correlation.unstack().dropna()
    unique_pairs = pairs.loc[pairs.index.get_level_values(0) < pairs.index.get_level_values(1)]
    sorted_pairs = unique_pairs.abs().sort_values(ascending=False)

    return sorted_pairs

def compare_distributions_and_stats(df1: pd.DataFrame, df2: pd.DataFrame, feature: str):
    feature1 = df1.loc[:, feature]
    feature2 = df2.loc[:, feature]
    
    train_skewness = round(sp.skew(feature1), 2)
    train_kurtosis = round(sp.kurtosis(feature1), 2)

    test_skewness = round(sp.skew(feature2), 2)
    test_kurtosis = round(sp.kurtosis(feature2), 2)
    
    feature_df = pd.concat([
        feature1.to_frame(feature).assign(dataset="Train"), 
        feature2.to_frame(feature).assign(dataset="Test")], 
        ignore_index=True)
    
    _, _ = plt.subplots(figsize=(9, 7))
    sns.histplot(feature_df, x=feature,  hue="dataset", kde=True, stat="density", linewidth=0, alpha=0.5, legend=True)
    plt.title(f"{feature}: Skewness (Train, Test) = ({train_skewness}, {test_skewness}) & Kurtosis (Train, Test) = ({train_kurtosis}, {test_kurtosis})")