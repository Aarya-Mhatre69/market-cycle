"""
Model wrappers for the stock clustering pipeline.
Exposes a consistent interface: fit(), predict(), save(), load().
"""

import logging
from typing import Any
import joblib
from pathlib import Path

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

class BaseModel:
    def fit(self, X: Any, y: Any = None) -> 'BaseModel':
        raise NotImplementedError
        
    def predict(self, X: Any) -> Any:
        raise NotImplementedError
        
    def save(self, filepath: str | Path) -> None:
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load(self, filepath: str | Path) -> None:
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")


class KMeansModel(BaseModel):
    def __init__(self, n_clusters: int, random_state: int = 42, n_init: int = 10, **kwargs):
        self.model = KMeans(
            n_clusters=n_clusters, 
            random_state=random_state, 
            n_init=n_init, 
            **kwargs
        )
        
    def fit(self, X: Any, y: Any = None) -> 'KMeansModel':
        self.model.fit(X)
        return self
        
    def predict(self, X: Any) -> Any:
        return self.model.predict(X)


class AgglomerativeModel(BaseModel):
    def __init__(self, n_clusters: int, metric: str = "precomputed", linkage: str = "average", **kwargs):
        self.model:AgglomerativeClustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=metric,
            linkage=linkage,
            **kwargs
        )
        
    def fit(self, X: Any, y: Any = None) -> 'AgglomerativeModel':
        # Agglomerative clustering does not have a separate predict method in the same way,
        # but we can fit it.
        self.model.fit(X)
        return self
        
    def predict(self, X: Any) -> Any:
        # AgglomerativeClustering typically uses fit_predict
        return self.model.fit_predict(X)


class IsolationForestModel(BaseModel):
    def __init__(self, contamination: float = 0.10, random_state: int = 42, **kwargs):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            **kwargs
        )
        
    def fit(self, X: Any, y: Any = None) -> 'IsolationForestModel':
        self.model.fit(X)
        return self
        
    def predict(self, X: Any) -> Any:
        return self.model.predict(X)
