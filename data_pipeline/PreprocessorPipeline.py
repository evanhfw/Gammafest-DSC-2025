from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._set_output import _SetOutputMixin
import pandas as pd


class PreprocessorPipeline(BaseEstimator, TransformerMixin, _SetOutputMixin):
    def __init__(self, metadata_df: pd.DataFrame):
        super().__init__()
        self.metadata_df = metadata_df
        self._sklearn_output_config = {"transform": "pandas"}

    def fit(self, x: pd.DataFrame, y=None) -> "PreprocessorPipeline":
        # x parameter required by sklearn API
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        x = x.merge(
            self.metadata_df, left_on="paper", right_on="paper_id", how="left"
        ).drop(columns=["paper_id"])
        x = x.merge(
            self.metadata_df,
            left_on="referenced_paper",
            right_on="paper_id",
            how="left",
            suffixes=("_original", "_referenced"),
        ).drop(columns=["paper_id"])
        x["publication_date_original"] = pd.to_datetime(
            x["publication_date_original"], format="mixed", errors="coerce"
        )
        x["publication_date_referenced"] = pd.to_datetime(
            x["publication_date_referenced"], format="mixed", errors="coerce"
        )
        return x
