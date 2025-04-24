"""Module containing pipeline for preprocessing paper metadata before feature engineering."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._set_output import _SetOutputMixin
import pandas as pd


class PreprocessorPipeline(BaseEstimator, TransformerMixin, _SetOutputMixin):
    """Pipeline for preprocessing paper citation data with metadata.

    Merges paper metadata with citation information and performs
    data type conversions for dates to prepare for feature engineering.
    """

    def __init__(self, metadata_df: pd.DataFrame):
        """Initialize the preprocessor pipeline with paper metadata.

        Args:
            metadata_df: DataFrame containing paper metadata to merge with citation data.
        """
        super().__init__()
        self.metadata_df = metadata_df
        self._sklearn_output_config = {"transform": "pandas"}

    def fit(self, x: pd.DataFrame, y=None) -> "PreprocessorPipeline":
        """Fit the transformer (no-op as this transformer is stateless).

        Args:
            x: Input features dataframe.
            y: Target values (unused).

        Returns:
            Self instance.
        """
        # x parameter required by sklearn API
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data by merging with metadata and converting date formats.

        Args:
            x: Input citation dataframe with paper and referenced_paper columns.

        Returns:
            DataFrame with merged metadata and converted date columns.
        """
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
