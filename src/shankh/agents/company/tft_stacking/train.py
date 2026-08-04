"""
Complete PyTorch Lightning Orchestration Driver for the Stacking TFT Pipeline.
Includes dynamic namespace alignment for lightning.pytorch vs pytorch_lightning.
"""
import json
import logging
from pathlib import Path
import joblib
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from shankh.agents.company.tft_stacking.config import CONFIG
from shankh.agents.company.tft_stacking.dataset import prepare_tft_frame, create_tft_datasets
from shankh.agents.company.tft_stacking.model import build_tft_model
from shankh.agents.company.tft_stacking.eval import evaluate_tft
from shankh.agents.company.data_loader import load_ohlcv
from shankh.agents.company.features import add_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_tft_pipeline")


def run_tft_pipeline() -> dict:
    """
    Executes end-to-end TFT training, artifact persistence, and test set evaluation.
    """
    pl.seed_everything(CONFIG["seed"])
    art_dir = Path(CONFIG["artifacts"]["artifacts_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("STEP 1 — Ingesting OHLCV Universe Panel")
    raw_df = load_ohlcv(CONFIG["data"]["data_dir"], required_cols=CONFIG["data"]["required_cols"])

    logger.info("STEP 2 — Engineering Core Technical Features")
    feat_df = add_features(raw_df, cfg=CONFIG.get("features"))

    logger.info("STEP 3 — Generating Layer 1 Meta-Features & Time-Index Frame")
    tft_frame = prepare_tft_frame(feat_df, CONFIG)

    logger.info("STEP 4 — Building PyTorch Forecasting TimeSeriesDataSet Objects")
    training_ds, validation_ds, pretrain_df, test_df = create_tft_datasets(tft_frame, CONFIG)

    train_dataloader = training_ds.to_dataloader(
        train=True,
        batch_size=CONFIG["trainer"]["batch_size"],
        num_workers=CONFIG["trainer"]["num_workers"],
    )
    val_dataloader = validation_ds.to_dataloader(
        train=False,
        batch_size=CONFIG["trainer"]["batch_size"] * 2,
        num_workers=CONFIG["trainer"]["num_workers"],
    )

    logger.info("STEP 5 — Instantiating Temporal Fusion Transformer Neural Architecture")
    tft_model = build_tft_model(training_ds, CONFIG)
    tft_model = torch.compile(tft_model,backend="inductor")

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=art_dir,
        filename="tft_best_checkpoint",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=CONFIG["trainer"]["early_stopping_patience"],
        verbose=True,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    tb_logger = TensorBoardLogger(save_dir=str(art_dir), name="tft_logs")

    trainer = pl.Trainer(
        max_epochs=CONFIG["trainer"]["max_epochs"],
        accelerator=CONFIG["trainer"]["accelerator"],
        gradient_clip_val=CONFIG["trainer"]["gradient_clip_val"],
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        logger=tb_logger,
    )

    logger.info("STEP 6 — Training TFT via PyTorch Lightning Trainer...")
    trainer.fit(
        tft_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    logger.info("STEP 7 — Persisting Dataset Parameters & Model Artifacts")
    joblib.dump(training_ds.get_parameters(), art_dir / CONFIG["artifacts"]["dataset_parameters"])

    # Load best checkpoint
    best_tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_cb.best_model_path)

    logger.info("STEP 8 — Running Evaluation on Held-Out Test Dataset")
    test_ds = TimeSeriesDataSet.from_dataset(training_ds, test_df, predict=True, stop_randomization=True)
    report = evaluate_tft(best_tft, test_ds, test_df, CONFIG)

    # Save JSON report
    with open(art_dir / CONFIG["artifacts"]["eval_report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 80)
    logger.info("STACKING TFT PIPELINE COMPLETE. Artifacts stored in: %s", art_dir)
    logger.info("=" * 80)
    return report


if __name__ == "__main__":
    run_tft_pipeline()