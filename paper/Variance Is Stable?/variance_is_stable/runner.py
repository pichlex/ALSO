from __future__ import annotations

import logging
from pathlib import Path

from .data import build_experiment_data, seed_everything
from .experiment_config import ExperimentConfig
from .metrics import clone_named_parameters, compute_epoch_start_metrics
from .modeling import build_model, resolve_device
from .results import save_run_artifacts
from .trainer import (
    build_epoch_record,
    build_optimizer,
    build_scheduler,
    evaluate,
    log_epoch_summary,
    parameter_shift_sq,
    train_one_epoch,
)


def run_experiment(
    config: ExperimentConfig,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    seed_everything(config.seed)
    device = resolve_device(config.device, logger=logger)
    logger.info("Starting run with config: %s", config.to_dict())
    logger.info("Resolved device: %s", device)

    data = build_experiment_data(config, device=device)
    logger.info(
        "Prepared CIFAR10 split: train=%s, val=%s, fixed first batch local indices=%s",
        len(data.train_dataset),
        len(data.val_dataset),
        data.train_order[: config.batch_size],
    )

    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    epoch_records: list[dict[str, float]] = []
    previous_epoch_start = None
    previous_hat_grad = None

    for epoch in range(config.epochs):
        logger.info("Epoch %s/%s: computing epoch-start metrics.", epoch + 1, config.epochs)
        current_epoch_start = clone_named_parameters(model)
        metric_3 = parameter_shift_sq(previous_epoch_start, current_epoch_start)
        epoch_metrics = compute_epoch_start_metrics(
            model=model,
            train_loader=data.train_loader,
            device=device,
            metric_microbatch_size=config.metric_microbatch_size,
            reference_grad=previous_hat_grad,
            progress_desc=f"Metrics {epoch + 1}/{config.epochs}",
        )

        logger.info("Epoch %s/%s: training.", epoch + 1, config.epochs)
        train_loss, current_hat_grad = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=data.train_loader,
            device=device,
            epoch_index=epoch,
            total_epochs=config.epochs,
        )

        if (epoch + 1) % config.eval_every == 0:
            val_loss, val_acc = evaluate(model, data.val_loader, device)
        else:
            val_loss = float("nan")
            val_acc = float("nan")

        if scheduler is not None:
            scheduler.step()

        epoch_record = build_epoch_record(
            epoch=epoch,
            epoch_metrics=epoch_metrics,
            metric_3=metric_3,
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
        )
        epoch_records.append(epoch_record)
        log_epoch_summary(logger, epoch, config.epochs, epoch_record)

        previous_epoch_start = current_epoch_start
        previous_hat_grad = current_hat_grad

    save_run_artifacts(output_dir=output_dir, config=config, epoch_records=epoch_records)
    logger.info("Finished run. Artifacts saved to %s", output_dir)
