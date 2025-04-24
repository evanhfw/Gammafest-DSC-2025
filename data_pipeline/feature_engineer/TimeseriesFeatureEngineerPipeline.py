"""Module containing pipeline for creating time-based features from publication dates."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._set_output import _SetOutputMixin
import pandas as pd


class TimeseriesFeatureEngineerPipeline(
    BaseEstimator, TransformerMixin, _SetOutputMixin
):
    """Pipeline for engineering features related to publication dates and years.

    Generates time-based features from publication dates, including year differences
    and temporal relationship indicators between original and referenced papers.
    """

    def __init__(self):
        """Initialize the pipeline with default output configuration."""
        super().__init__()
        self.set_output(transform="default")

    def set_output(self, *, transform=None):
        """Set the output configuration for the transformer.

        Args:
            transform: Output format configuration.

        Returns:
            Self instance.
        """
        self._sklearn_output_config = {"transform": transform}
        return self

    def fit(self, x: pd.DataFrame, y=None) -> "TimeseriesFeatureEngineerPipeline":
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
        """Transform the input data to generate time-based features.

        Args:
            x: Input features dataframe.

        Returns:
            DataFrame with engineered time-based features.
        """
        x = x.copy()
        x["year_difference"] = (
            x["publication_year_original"] - x["publication_year_referenced"]
        )
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
