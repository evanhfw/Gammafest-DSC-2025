"""Module containing pipeline for creating features based on citation counts."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._set_output import _SetOutputMixin
import pandas as pd


class CitedCountFeatureEngineerPipeline(
    BaseEstimator, TransformerMixin, _SetOutputMixin
):
    """Pipeline for engineering features related to citation counts.

    Generates features that compare citation counts between original and referenced papers,
    including citation differences and positive difference indicators.
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

    def fit(self, x: pd.DataFrame, y=None) -> "CitedCountFeatureEngineerPipeline":
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
        """Transform the input data to generate citation-based features.

        Args:
            x: Input features dataframe.

        Returns:
            DataFrame with engineered citation-based features.
        """
        x = x.copy()
        x["cited_by_count_difference"] = (
            x["cited_by_count_original"] - x["cited_by_count_referenced"]
        )
        x["positive_cited_by_count_difference"] = (
            x["cited_by_count_difference"] >= 0
        ).astype(int)
        return x[["cited_by_count_difference", "positive_cited_by_count_difference"]]
