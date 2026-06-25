from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ValidationDataset:
    """
    Lightweight metadata object for an independent validation dataset.

    This object describes the dataset.
    It does not store the GeoDataFrame.
    """

    name: str
    path: str
    geometry_type: str
    source: str
    metrics: List[str]
    weight: float = 1.0
    id_column: Optional[str] = None
    category: Optional[str] = None
    independent_of_model: bool = True
    notes: str = ""

    def validate_metadata(self) -> None:
        allowed_geometry = {
            "point",
            "line",
            "polygon",
            "mixed",
            "point_or_polygon",
        }

        allowed_metrics = {
            "intersects",
            "overlap",
            "distance",
            "density",
        }

        if self.geometry_type not in allowed_geometry:
            raise ValueError(
                f"Invalid geometry_type for {self.name}: {self.geometry_type}"
            )

        unknown_metrics = set(self.metrics) - allowed_metrics
        if unknown_metrics:
            raise ValueError(
                f"Invalid metrics for {self.name}: {unknown_metrics}"
            )

        if self.weight < 0:
            raise ValueError(
                f"Dataset weight must be nonnegative for {self.name}"
            )
