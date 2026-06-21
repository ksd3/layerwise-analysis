"""Base class for all data sources."""

import os
import glob
from abc import ABC, abstractmethod

import pandas as pd

from ..config import PipelineConfig
from ..utils.parquet import save_shard, load_shards


class DataSource(ABC):
    """Abstract base for a data source (spectra, images, light curves, etc.).

    Subclasses implement fetch() to download/process data and write shards.
    Caching, preflight, and shard management are handled by the base class.
    """

    name: str  # e.g. "apogee", "ps1", "ztf"

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.shard_dir = os.path.join(config.work_dir, "shards", self.name)

    def is_cached(self) -> bool:
        """Check if shards already exist for this source."""
        return (os.path.isdir(self.shard_dir)
                and bool(glob.glob(os.path.join(self.shard_dir, "*.parquet"))))

    def preflight(self) -> tuple[bool, str]:
        """Check if this source is reachable/available. Override in subclasses."""
        return True, "No check defined"

    @abstractmethod
    def fetch(self, catalog_df: pd.DataFrame) -> int:
        """Fetch data for catalog objects, write shards. Return match count."""
        ...

    def save_shard(self, df: pd.DataFrame, shard_idx: int) -> str:
        """Write a shard for this source."""
        return save_shard(df, self.shard_dir, shard_idx)

    def load_shards(self) -> pd.DataFrame | None:
        """Load all shards for this source."""
        return load_shards(self.shard_dir)

    def run(self, catalog_df: pd.DataFrame) -> int:
        """Run with caching: skip if shards exist, otherwise fetch."""
        if self.is_cached():
            print(f"    {self.name}: already completed (shards exist), skipping")
            return 0
        return self.fetch(catalog_df)
