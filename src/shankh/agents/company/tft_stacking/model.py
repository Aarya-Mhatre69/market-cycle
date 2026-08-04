"""
Model builder for the Temporal Fusion Transformer architecture.
"""
import logging
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import TimeSeriesDataSet
logger = logging.getLogger(__name__)


def build_tft_model(
    training_dataset: TimeSeriesDataSet,
    cfg: dict,
) -> TemporalFusionTransformer:
    """
    Instantiates a TemporalFusionTransformer with QuantileLoss.
    """
    m_cfg = cfg["model"]
    quantiles = m_cfg.get("quantiles", [0.16, 0.84])

    tft:TemporalFusionTransformer = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=m_cfg["learning_rate"],
        hidden_size=m_cfg["hidden_size"],
        attention_head_size=m_cfg["attention_head_size"],
        dropout=m_cfg["dropout"],
        hidden_continuous_size=m_cfg["hidden_continuous_size"],
        output_size=len(quantiles),
        loss=QuantileLoss(quantiles=quantiles),
        log_interval=10,
        reduce_on_plateau_patience=m_cfg["reduce_on_plateau_patience"],
        weight_decay=m_cfg["weight_decay"],
    )
    logger.info(
        "TFT Model Configured | Hidden Size: %d | Attention Heads: %d | Quantiles: %s",
        m_cfg["hidden_size"],
        m_cfg["attention_head_size"],
        quantiles,
    )
    return tft