"""
src/shankh/analytics/cluster_trainer.py
End-to-End Stock Clustering & Anomaly Detection Training Pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


class StockClusterTrainer:
    """
    Extracts multi-factor financial features from OHLCV data, computes return correlation
    distances, and trains Clustering & Forensic Anomaly Detection models.
    """

    def __init__(self, n_clusters: int = 5, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = RobustScaler()

    # ---------------------------------------------------------------------------
    # Feature Engineering Pipeline
    # ---------------------------------------------------------------------------
    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extracts both time-series price returns and multi-factor feature profiles per ticker.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (pivoted_prices_df, factor_features_df)
        """
        logger.info("Extracting stock features for clustering...")
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        # 1. Pivot price matrix for correlation distance
        pivoted_prices = df.pivot(index="date", columns="ticker", values="close")

        # 2. Derive equal-weighted market return as benchmark for Beta calculation
        df["log_return"] = df.groupby("ticker")["close"].transform(lambda x: np.log(x / x.shift(1)))
        mkt_returns = df.groupby("date")["log_return"].mean().rename("mkt_return")
        df = df.merge(mkt_returns, on="date", how="left")

        # Feature aggregation dictionary per ticker
        features_list = []

        for ticker, t_df in df.groupby("ticker"):
            t_df = t_df.dropna(subset=["log_return"]).copy()
            if len(t_df) < 60:
                continue

            returns = t_df["log_return"].values
            mkt_ret = t_df["mkt_return"].values

            # Statistical / Return Metrics
            ann_return = float(np.mean(returns) * 252)
            ann_vol = float(np.std(returns) * np.sqrt(252))
            sharpe = (ann_return - 0.065) / (ann_vol + 1e-6)  # 6.5% RBI repo rate baseline
            skewness = float(pd.Series(returns).skew())

            # Max Drawdown
            cum_rets = np.exp(np.cumsum(returns))
            peak = np.maximum.accumulate(cum_rets)
            drawdown = (cum_rets - peak) / peak
            max_dd = float(np.min(drawdown))

            # Beta calculation vs Market
            cov = np.cov(returns, mkt_ret)
            beta = float(cov[0, 1] / (cov[1, 1] + 1e-8))

            # Technical Ratios (RSI, ATR / Price ratio, 20-DMA trend)
            close = t_df["close"].values
            high = t_df["high"].values
            low = t_df["low"].values

            # RSI 14
            delta = pd.Series(close).diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-8)
            rsi = float((100 - (100 / (1 + rs))).dropna().iloc[-1]) if len(rs.dropna()) > 0 else 50.0

            # Normalized ATR (ATR / Close)
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = float(pd.Series(tr).rolling(14).mean().dropna().iloc[-1]) if len(tr) >= 14 else 0.0
            atr_pct = atr / (close[-1] + 1e-8)

            # Price relative to 20-DMA
            sma20 = float(pd.Series(close).rolling(20).mean().iloc[-1])
            dist_20dma = (close[-1] - sma20) / sma20

            features_list.append({
                "ticker": ticker,
                "ann_return": ann_return,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "beta": beta,
                "skewness": skewness,
                "max_drawdown": max_dd,
                "rsi_14": rsi,
                "atr_pct": atr_pct,
                "dist_20dma": dist_20dma,
            })

        factor_df = pd.DataFrame(features_list).set_index("ticker").dropna()
        logger.info("Successfully extracted features for %d tickers.", len(factor_df))
        return pivoted_prices[factor_df.index], factor_df

    # ---------------------------------------------------------------------------
    # Model Training & Evaluation
    # ---------------------------------------------------------------------------
    def train(
        self,
        pivoted_prices: pd.DataFrame,
        factor_df: pd.DataFrame,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """
        Trains Correlation Clustering, Multi-Factor K-Means, and Isolation Forest Outlier Detection.
        Saves models and artifacts to output_dir.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Correlation Distance Agglomerative Clustering
        returns_df = np.log(pivoted_prices / pivoted_prices.shift(1)).dropna()
        corr_matrix = returns_df.corr().clip(-1.0, 1.0)
        dist_matrix = np.sqrt(2 * (1 - corr_matrix.values))

        corr_cluster_model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric="precomputed",
            linkage="average",
        )
        corr_labels = corr_cluster_model.fit_predict(dist_matrix)
        corr_assignments = {ticker: int(label) for ticker, label in zip(factor_df.index, corr_labels)}

        # 2. Multi-Factor K-Means Clustering
        scaled_factors = self.scaler.fit_transform(factor_df)
        kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        kmeans_labels = kmeans_model.fit_predict(scaled_factors)
        kmeans_assignments = {ticker: int(label) for ticker, label in zip(factor_df.index, kmeans_labels)}

        # Quality Metrics
        sil_score = float(silhouette_score(scaled_factors, kmeans_labels))
        db_score = float(davies_bouldin_score(scaled_factors, kmeans_labels))
        ch_score = float(calinski_harabasz_score(scaled_factors, kmeans_labels))

        # 3. Forensic Anomaly Detection (Isolation Forest)
        iso_forest = IsolationForest(contamination=0.10, random_state=self.random_state)
        anomaly_preds = iso_forest.fit_predict(scaled_factors)
        anomalies = [ticker for ticker, pred in zip(factor_df.index, anomaly_preds) if pred == -1]

        # Structure Export JSON
        artifacts_metadata = {
            "n_clusters": self.n_clusters,
            "tickers_count": len(factor_df),
            "evaluation_metrics": {
                "silhouette_score": round(sil_score, 4),
                "davies_bouldin_index": round(db_score, 4),
                "calinski_harabasz_index": round(ch_score, 4),
            },
            "correlation_clusters": corr_assignments,
            "kmeans_factor_clusters": kmeans_assignments,
            "flagged_forensic_anomalies": anomalies,
            "feature_columns": list(factor_df.columns),
        }

        # Save Artifacts
        joblib.dump(self.scaler, output_dir / "cluster_scaler.joblib")
        joblib.dump(kmeans_model, output_dir / "kmeans_cluster_model.joblib")
        joblib.dump(iso_forest, output_dir / "isolation_forest_model.joblib")

        with open(output_dir / "cluster_results.json", "w", encoding="utf-8") as f:
            json.dump(artifacts_metadata, f, indent=2)

        logger.info(
            "Saved Clustering Artifacts to %s | Silhouette: %.3f | Anomalies Flagged: %d",
            output_dir, sil_score, len(anomalies)
        )
        return artifacts_metadata


"""
src/shankh/drivers/drive_clustering.py
Driver script to run, evaluate, and inspect Stock Clustering & Forensic Anomaly Detection.
"""

import json
import logging
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
from shankh.price_band.data_loader import load_ohlcv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drive_clustering")


def run_clustering_driver():
    data_dir = PROJECT_ROOT / "data" / "universe"
    output_dir = PROJECT_ROOT / "models" / "clustering"

    print("=" * 80)
    print(" SHANKH ANALYTICS DRIVER: STOCK CLUSTERING & FORENSIC ANOMALY DETECTION")
    print("=" * 80)
    print(f" Data Directory   : {data_dir}")
    print(f" Artifacts Output : {output_dir}")
    print("-" * 80)

    # 1. Load Real Data from Universe
    logger.info("Loading ticker data from universe CSVs...")
    df = load_ohlcv(
        data_dir=data_dir,
        min_rows=500,
        min_tickers=15,
        max_tickers=40,
        min_overlap_days=500,
    )

    # 2. Instantiate and Extract Features
    trainer = StockClusterTrainer(n_clusters=4, random_state=42)
    pivoted_prices, factor_df = trainer.extract_features(df)

    print("\n" + "=" * 80)
    print(" EXTRACTED MULTI-FACTOR FEATURE SUMMARY (Sample)")
    print("=" * 80)
    print(factor_df.head(10).to_string())

    # 3. Train Models and Save Artifacts
    logger.info("Training Clustering & Anomaly Detection Models...")
    results = trainer.train(
        pivoted_prices=pivoted_prices,
        factor_df=factor_df,
        output_dir=output_dir,
    )

    # 4. Display Evaluation & Cluster Groupings Report
    print("\n" + "=" * 80)
    print(" MODEL EVALUATION METRICS")
    print("=" * 80)
    metrics = results["evaluation_metrics"]
    print(f"  - Silhouette Score       : {metrics['silhouette_score']:.4f}  (Higher = better separated clusters)")
    print(f"  - Davies-Bouldin Index   : {metrics['davies_bouldin_index']:.4f}  (Lower = tighter clusters)")
    print(f"  - Calinski-Harabasz Index: {metrics['calinski_harabasz_index']:.2f}")

    print("\n" + "=" * 80)
    print(" CLUSTER ASSIGNMENTS (Multi-Factor K-Means)")
    print("=" * 80)
    grouped_clusters = {}
    for tkr, c_id in results["kmeans_factor_clusters"].items():
        grouped_clusters.setdefault(c_id, []).append(tkr)

    for cluster_id, tickers in sorted(grouped_clusters.items()):
        print(f"\n [Cluster {cluster_id}] ({len(tickers)} Stocks):")
        print(f"   {', '.join(tickers)}")
        # Print cluster centroid statistics
        c_stats = factor_df.loc[tickers].mean()
        print(f"   Mean Beta: {c_stats['beta']:.2f} | Ann Vol: {c_stats['ann_vol']*100:.1f}% | Sharpe: {c_stats['sharpe']:.2f} | RSI: {c_stats['rsi_14']:.1f}")

    print("\n" + "=" * 80)
    print(" FORENSIC ANOMALY SCREENING (Isolation Forest)")
    print("=" * 80)
    anomalies = results["flagged_forensic_anomalies"]
    if anomalies:
        print(f" Flagged {len(anomalies)} Outlier Stock(s) Requiring Human Review:")
        for anom in anomalies:
            stats = factor_df.loc[anom]
            print(f"  -> {anom:15s} | Vol: {stats['ann_vol']*100:5.1f}% | Beta: {stats['beta']:5.2f} | Max DD: {stats['max_drawdown']*100:5.1f}%")
    else:
        print("  No extreme structural outliers flagged in this sample universe.")

    print("\n" + "=" * 80)
    print(" DRIVER RUN COMPLETE: Artifacts saved to 'models/clustering/'")
    print("=" * 80)


if __name__ == "__main__":
    run_clustering_driver()