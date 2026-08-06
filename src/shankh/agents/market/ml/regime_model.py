"""
Model wrappers for the market regime pipeline.
Exposes a consistent interface: fit(), predict(), predict_proba(), save(), load().
"""
import logging
from pathlib import Path
from typing import Any
import joblib
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)


class BaseModel:
    def fit(self, X: Any, y: Any = None) -> 'BaseModel':
        raise NotImplementedError
        
    def predict(self, X: Any) -> Any:
        raise NotImplementedError
        
    def save(self, filepath: str | Path) -> None:
        joblib.dump(self.model, filepath)
        logger.info("Model saved to %s", filepath)
        
    def load(self, filepath: str | Path) -> None:
        self.model = joblib.load(filepath)
        logger.info("Model loaded from %s", filepath)


class HMMModel(BaseModel):
    def __init__(self, n_regimes: int, random_state: int = 42, covariance_type: str = "full", n_iter: int = 100, **kwargs):
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            **kwargs
        )
        
    def fit(self, X: Any, y: Any = None) -> 'HMMModel':
        self.model.fit(X)
        return self
        
    def predict(self, X: Any) -> Any:
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self.model.predict_proba(X)