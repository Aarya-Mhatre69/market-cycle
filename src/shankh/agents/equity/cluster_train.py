"""
Pipeline entrypoint for Stock Clustering & Anomaly Detection Training.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler

from shankh.agents.equity.cluster_config import CONFIG
from shankh.agents.equity.cluster_data_loader import load_data
from shankh.agents.equity.cluster_features import build_cluster_features
from shankh.agents.equity.cluster_validation import validate_features
from shankh.agents.equity.cluster_model import KMeansModel, AgglomerativeModel, IsolationForestModel
from shankh.agents.equity.cluster_evaluation import evaluate_clustering

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_cluster")

def train_pipeline() -> Dict[str, Any]:
    """
    End-to-End training flow.
    """
    data_dir = Path(CONFIG["data"]["data_dir"])
    output_dir = Path(CONFIG["artifacts"]["models_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" SHANKH ANALYTICS DRIVER: STOCK CLUSTERING & FORENSIC ANOMALY DETECTION")
    print("=" * 80)
    print(f" Data Directory   : {data_dir}")
    print(f" Artifacts Output : {output_dir}")
    print("-" * 80)

    # 1. Load Data
    df = load_data(
        data_dir=data_dir,
        min_rows=CONFIG["data"]["min_rows"],
        min_tickers=CONFIG["data"]["min_tickers"],
        max_tickers=CONFIG["data"]["max_tickers"],
        min_overlap_days=CONFIG["data"]["min_overlap_days"],
        required_cols=CONFIG["data"]["required_cols"],
    )

    # 2. Engineer Features
    pivoted_prices, factor_df = build_cluster_features(df, CONFIG["features"])

    # 3. Validate
    if not validate_features(factor_df, pivoted_prices):
        raise ValueError("Validation failed on extracted features.")

    print("\n" + "=" * 80)
    print(" EXTRACTED MULTI-FACTOR FEATURE SUMMARY (Sample)")
    print("=" * 80)
    print(factor_df.head(10).to_string())

    logger.info("Training Clustering & Anomaly Detection Models...")
    
    # 4. Train Models
    returns_df = np.log(pivoted_prices / pivoted_prices.shift(1)).dropna()
    corr_matrix = returns_df.corr().clip(-1.0, 1.0)
    dist_matrix = np.sqrt(2 * (1 - corr_matrix.values))

    corr_cluster_model = AgglomerativeModel(
        n_clusters=CONFIG["model"]["n_clusters"],
        **CONFIG["model"]["agglomerative"]
    )
    corr_labels = corr_cluster_model.predict(dist_matrix)
    corr_assignments = {ticker: int(label) for ticker, label in zip(factor_df.index, corr_labels)}

    scaler = RobustScaler()
    scaled_factors = scaler.fit_transform(factor_df)
    
    kmeans_model = KMeansModel(
        n_clusters=CONFIG["model"]["n_clusters"],
        random_state=CONFIG["seed"],
        **CONFIG["model"]["kmeans"]
    )
    kmeans_model.fit(scaled_factors)
    kmeans_labels = kmeans_model.predict(scaled_factors)
    kmeans_assignments = {ticker: int(label) for ticker, label in zip(factor_df.index, kmeans_labels)}

    iso_forest = IsolationForestModel(
        random_state=CONFIG["seed"],
        **CONFIG["model"]["isolation_forest"]
    )
    iso_forest.fit(scaled_factors)
    anomaly_preds = iso_forest.predict(scaled_factors)
    anomalies = [ticker for ticker, pred in zip(factor_df.index, anomaly_preds) if pred == -1]

    metrics = evaluate_clustering(scaled_factors, kmeans_labels)

    artifacts_metadata = {
        "n_clusters": CONFIG["model"]["n_clusters"],
        "tickers_count": len(factor_df),
        "evaluation_metrics": metrics,
        "correlation_clusters": corr_assignments,
        "kmeans_factor_clusters": kmeans_assignments,
        "flagged_forensic_anomalies": anomalies,
        "feature_columns": list(factor_df.columns),
    }

    joblib.dump(scaler, output_dir / CONFIG["artifacts"]["cluster_scaler"])
    kmeans_model.save(output_dir / CONFIG["artifacts"]["kmeans_model"])
    iso_forest.save(output_dir / CONFIG["artifacts"]["isolation_forest"])

    with open(output_dir / CONFIG["artifacts"]["cluster_results"], "w", encoding="utf-8") as f:
        json.dump(artifacts_metadata, f, indent=2)

    logger.info(
        "Saved Clustering Artifacts to %s | Silhouette: %.3f | Anomalies Flagged: %d",
        output_dir, metrics["silhouette_score"], len(anomalies)
    )

    print("\n" + "=" * 80)
    print(" MODEL EVALUATION METRICS")
    print("=" * 80)
    print(f"  - Silhouette Score       : {metrics['silhouette_score']:.4f}  (Higher = better separated clusters)")
    print(f"  - Davies-Bouldin Index   : {metrics['davies_bouldin_index']:.4f}  (Lower = tighter clusters)")
    print(f"  - Calinski-Harabasz Index: {metrics['calinski_harabasz_index']:.2f}")

    print("\n" + "=" * 80)
    print(" CLUSTER ASSIGNMENTS (Multi-Factor K-Means)")
    print("=" * 80)
    grouped_clusters = {}
    for tkr, c_id in kmeans_assignments.items():
        grouped_clusters.setdefault(c_id, []).append(tkr)

    for cluster_id, tickers in sorted(grouped_clusters.items()):
        print(f"\n [Cluster {cluster_id}] ({len(tickers)} Stocks):")
        print(f"   {', '.join(tickers)}")
        c_stats = factor_df.loc[tickers].mean()
        print(f"   Mean Beta: {c_stats['beta']:.2f} | Ann Vol: {c_stats['ann_vol']*100:.1f}% | Sharpe: {c_stats['sharpe']:.2f} | RSI: {c_stats['rsi_14']:.1f}")

    print("\n" + "=" * 80)
    print(" FORENSIC ANOMALY SCREENING (Isolation Forest)")
    print("=" * 80)
    if anomalies:
        print(f" Flagged {len(anomalies)} Outlier Stock(s) Requiring Human Review:")
        for anom in anomalies:
            stats = factor_df.loc[anom]
            print(f"  -> {anom:15s} | Vol: {stats['ann_vol']*100:5.1f}% | Beta: {stats['beta']:5.2f} | Max DD: {stats['max_drawdown']*100:5.1f}%")
    else:
        print("  No extreme structural outliers flagged in this sample universe.")

    print("\n" + "=" * 80)
    print(f" DRIVER RUN COMPLETE: Artifacts saved to '{output_dir}'")
    print("=" * 80)
    
    return artifacts_metadata


if __name__ == "__main__":
    train_pipeline()
