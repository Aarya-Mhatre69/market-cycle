"""
Meta-feature generator for loading Layer 1 model artifacts and creating 
stacked input features for the Layer 2 Temporal Fusion Transformer.
"""
import logging
from pathlib import Path
import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def generate_layer1_meta_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Loads pre-trained Layer 1 model artifacts from disk and merges their outputs
    into the feature frame. If an artifact is missing, safe default fallbacks are used.
    """
    df_work = df.copy()
    _SRC_ROOT = Path(__file__).resolve().parents[4]

    regime_dir = _SRC_ROOT / "models" / "regime" / "artifacts"
    cluster_dir = _SRC_ROOT / "models" / "clustering" / "artifacts"
    price_band_dir = _SRC_ROOT / "models" / "price_band" / "pyro_glu" / "artifacts"

    # 1. Layer 1a: Gaussian HMM (Market Regime)
    try:
        if (regime_dir / "regime_model.joblib").exists():
            from shankh.ml.market.features import build_regime_features
            
            logger.info("Loading Layer 1a HMM artifacts from %s", regime_dir)
            hmm_model = joblib.load(regime_dir / "regime_model.joblib")
            hmm_scaler = joblib.load(regime_dir / "regime_scaler.joblib")

            regime_df = build_regime_features(df_work, cfg.get("features", {}))
            feature_cols_hmm = ["mkt_volatility", "breadth_pct_above_20dma", "correlation_density"]
            
            X_hmm = hmm_scaler.transform(regime_df[feature_cols_hmm])
            regime_probs = hmm_model.predict_proba(X_hmm)

            regime_meta = pd.DataFrame({
                "date": regime_df.index,
                "regime_state": hmm_model.predict(X_hmm).astype(str),
                "prob_risk_off": regime_probs[:, -1],
                "mkt_volatility": regime_df["mkt_volatility"].values,
            })
            df_work = df_work.merge(regime_meta, on="date", how="left")
        else:
            raise FileNotFoundError("HMM artifacts missing")
    except Exception as exc:
        logger.warning("Could not load Layer 1a HMM (%s). Using defaults.", exc)
        df_work["regime_state"] = "0"
        df_work["prob_risk_off"] = 0.10
        df_work["mkt_volatility"] = 15.0

    # 2. Layer 1b: K-Means & Isolation Forest (Clustering)
    try:
        if (cluster_dir / "kmeans_cluster_model.joblib").exists():
            from shankh.ml.macro.features import build_cluster_features
            
            logger.info("Loading Layer 1b Clustering artifacts from %s", cluster_dir)
            cluster_scaler = joblib.load(cluster_dir / "cluster_scaler.joblib")
            kmeans_model = joblib.load(cluster_dir / "kmeans_cluster_model.joblib")
            iso_forest_model = joblib.load(cluster_dir / "isolation_forest_model.joblib")

            pivoted_prices, factor_df = build_cluster_features(df_work, cfg.get("features", {}))
            X_cluster = cluster_scaler.transform(factor_df)

            cluster_meta = pd.DataFrame({
                "ticker": factor_df.index,
                "kmeans_cluster_id": kmeans_model.predict(X_cluster).astype(str),
                "anomaly_score": iso_forest_model.predict(X_cluster).astype(float),
            })
            df_work = df_work.merge(cluster_meta, on="ticker", how="left")
        else:
            raise FileNotFoundError("Clustering artifacts missing")
    except Exception as exc:
        logger.warning("Could not load Layer 1b Clustering (%s). Using defaults.", exc)
        df_work["kmeans_cluster_id"] = "0"
        df_work["anomaly_score"] = 1.0

    # 3. Layer 1c: LightGBM Boosters
    try:
        if (price_band_dir / "upper_model.txt").exists():
            import lightgbm as lgb
            logger.info("Loading Layer 1c LightGBM artifacts from %s", price_band_dir)
            upper_booster = lgb.Booster(model_file=str(price_band_dir / "upper_model.txt"))
            lower_booster = lgb.Booster(model_file=str(price_band_dir / "lower_model.txt"))
            feature_cols_lgb = joblib.load(price_band_dir / "feature_cols.pkl")

            X_lgb = df_work[feature_cols_lgb]
            df_work["lgb_pred_upper"] = upper_booster.predict(X_lgb)
            df_work["lgb_pred_lower"] = lower_booster.predict(X_lgb)
        else:
            raise FileNotFoundError("LightGBM boosters missing")
    except Exception as exc:
        logger.warning("Could not load Layer 1c LightGBM (%s). Using defaults.", exc)
        df_work["lgb_pred_upper"] = 0.02
        df_work["lgb_pred_lower"] = -0.02

    # Fill any remaining NaNs
    df_work["regime_state"] = df_work["regime_state"].fillna("0").astype(str)
    df_work["kmeans_cluster_id"] = df_work["kmeans_cluster_id"].fillna("0").astype(str)
    df_work["prob_risk_off"] = df_work["prob_risk_off"].fillna(0.10).astype(float)
    df_work["anomaly_score"] = df_work["anomaly_score"].fillna(1.0).astype(float)
    df_work["mkt_volatility"] = df_work["mkt_volatility"].fillna(15.0).astype(float)
    df_work["lgb_pred_upper"] = df_work["lgb_pred_upper"].fillna(0.02).astype(float)
    df_work["lgb_pred_lower"] = df_work["lgb_pred_lower"].fillna(-0.02).astype(float)

    return df_work