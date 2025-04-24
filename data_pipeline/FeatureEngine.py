from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._set_output import _SetOutputMixin
import pandas as pd


class TimeseriesFeatureEngineerPipeline(
    BaseEstimator, TransformerMixin, _SetOutputMixin
):
    def __init__(self):
        super().__init__()
        self._sklearn_output_config = {"transform": "pandas"}

    def fit(self, x: pd.DataFrame, y=None) -> "TimeseriesFeatureEngineerPipeline":
        # x parameter required by sklearn API
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        x["year_difference"] = x["year_original"] - x["year_referenced"]
        x["is_original_before_referenced"] = (
            x["publication_date_original"] > x["publication_date_referenced"]
        ).astype(int)
        x["positive_year_difference"] = (x["year_difference"] >= 0).astype(int)
        return x[
            [
                "year_difference",
                "is_original_before_referenced",
                "positive_year_difference",
            ]
        ]


class CitedCountFeatureEngineerPipeline(
    BaseEstimator, TransformerMixin, _SetOutputMixin
):
    def __init__(self):
        super().__init__()
        self._sklearn_output_config = {"transform": "pandas"}

    def fit(self, x: pd.DataFrame, y=None) -> "CitedCountFeatureEngineerPipeline":
        # x parameter required by sklearn API
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        x["cited_by_count_difference"] = (
            x["cited_by_count_original"] - X["cited_by_count_referenced"]
        )
        x["positive_cited_by_count_difference"] = (
            x["cited_by_count_difference"] >= 0
        ).astype(int)
        return x[["cited_by_count_difference", "positive_cited_by_count_difference"]]
