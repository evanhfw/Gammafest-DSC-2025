"""Module containing pipeline for creating features based on shared counts and similarity probabilities."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._set_output import _SetOutputMixin
import pandas as pd


class SharedCountProbabilityFeatureEngineerPipeline(
    BaseEstimator, TransformerMixin, _SetOutputMixin
):
    """Pipeline for engineering features related to shared concepts and authors.

    Calculates various similarity metrics between papers, including shared concept counts,
    concept similarity probabilities, shared author counts, and author similarity probabilities.
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

    def fit(
        self, x: pd.DataFrame, y=None
    ) -> "SharedCountProbabilityFeatureEngineerPipeline":
        """Fit the transformer (no-op as this transformer is stateless).

        Args:
            x: Input features dataframe.
            y: Target values (unused).

        Returns:
            Self instance.
        """
        # x parameter required by sklearn API
        return self

    def extract_items(self, item_str):
        """Extract individual items from a semicolon or comma-separated string.

        Args:
            item_str: String containing items separated by semicolons or commas.

        Returns:
            Set of lowercase, trimmed items.
        """
        if pd.isna(item_str):
            return set()
        # Split by semicolon and strip whitespace, make lowercase
        items = [item.strip().lower() for item in item_str.split(";")]
        # Also handle if items are separated by commas
        if len(items) == 1:
            items = [item.strip().lower() for item in item_str.split(",")]
        return set(items)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data to generate similarity features.

        Calculates shared concepts/authors counts and similarity probabilities
        between original and referenced papers.

        Args:
            x: Input features dataframe.

        Returns:
            DataFrame with engineered similarity features.
        """
        x = x.copy()

        x["shared_concept_count"] = 0
        x["concept_similarity_probability"] = 0.0
        x["shared_author_count"] = 0
        x["author_similarity_probability"] = 0.0

        shared_concept_counts = []
        concept_similarity_probs = []
        shared_author_counts = []
        author_similarity_probs = []

        for _, row in x.iterrows():
            # Process concepts
            original_concepts = self.extract_items(row["concepts_original"])
            referenced_concepts = self.extract_items(row["concepts_referenced"])

            shared_concepts = original_concepts.intersection(referenced_concepts)
            all_concepts = original_concepts.union(referenced_concepts)

            shared_concept_count = len(shared_concepts)
            shared_concept_counts.append(shared_concept_count)

            if len(all_concepts) > 0:
                concept_similarity_prob = len(shared_concepts) / len(all_concepts)
            else:
                concept_similarity_prob = 0.0
            concept_similarity_probs.append(concept_similarity_prob)

            # Process authors
            original_authors = self.extract_items(row["authors_original"])
            referenced_authors = self.extract_items(row["authors_referenced"])

            shared_authors = original_authors.intersection(referenced_authors)
            all_authors = original_authors.union(referenced_authors)

            shared_author_count = len(shared_authors)
            shared_author_counts.append(shared_author_count)

            if len(all_authors) > 0:
                author_similarity_prob = len(shared_authors) / len(all_authors)
            else:
                author_similarity_prob = 0.0
            author_similarity_probs.append(author_similarity_prob)

        # Update all values at once
        x["shared_concept_count"] = shared_concept_counts
        x["concept_similarity_probability"] = concept_similarity_probs
        x["shared_author_count"] = shared_author_counts
        x["author_similarity_probability"] = author_similarity_probs

        return x[
            [
                "shared_concept_count",
                "concept_similarity_probability",
                "shared_author_count",
                "author_similarity_probability",
            ]
        ]
