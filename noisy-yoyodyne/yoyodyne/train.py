"""Training."""

import os
import time
from typing import Dict, List, Optional, Tuple

import click
import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks, loggers
from torch.utils import data

from . import collators, datasets, evaluators, models, predict, util


def make_training_args(
    arch: str,
    train_set: data.Dataset,
    embedding_size: int,
    hidden_size: int,
    nhead: int,
    max_seq_len: int,
    optimizer: torch.optim.Optimizer,
    beta1: float,
    beta2: float,
    warmup_steps: int,
    learning_rate: float,
    evaluator: evaluators.Evaluator,
    max_decode_len: int,
    dropout: float,
    enc_layers: int,
    dec_layers: int,
    bidirectional: bool,
    label_smoothing: Optional[float],
    lr_scheduler: str,
    expert: Optional[models.expert.Expert],
) -> Dict:
    """Prepares the training args for the specific architecture.

    Args:
        arch (str): the architecture.
        train_set (data.Dataset): training dataset.
        embedding_size (int).
        hidden_size (int).
        nhead (int).
        max_seq_len (int): maximum input sequence length for transformer
            positional encoding.
        optimizer (optim.Optimizer).
        beta1 (float).
        beta2 (float).
        warmup_steps (int).
        learning_rate (float).
        evaluator (evaluators.Evaluator).
        max_decode_len (int).
        dropout (float).
        enc_layers (int).
        dec_layers (int).
        bidirectional (bool).
        label_smoothing (float).
        lr_scheduler (str).
        expert (OptimalSubstitutionExpert).

    Returns:
        Dict: Dictionary of arguments.
    """
    args = {
        "vocab_size": train_set.source_vocab_size,
        "features_vocab_size": getattr(train_set, "features_vocab_size", -1),
        "features_idx": getattr(train_set, "features_idx", -1),
        "embedding_size": embedding_size,
        "hidden_size": hidden_size,
        "output_size": train_set.target_vocab_size,
        "pad_idx": train_set.pad_idx,
        "start_idx": train_set.start_idx,
        "end_idx": train_set.end_idx,
        "optim": optimizer,
        "beta1": beta1,
        "beta2": beta2,
        "lr": learning_rate,
        "evaluator": evaluator,
        "max_decode_len": max_decode_len,
        "dropout": dropout,
        "enc_layers": enc_layers,
        "dec_layers": dec_layers,
        "label_smoothing": label_smoothing,
        "warmup_steps": warmup_steps,
        "scheduler": lr_scheduler,
        "bidirectional": bidirectional,
        "nhead": nhead,
        "max_seq_len": max_seq_len,
        "expert": expert,
    }
    return args


def make_pretraining_pl_callbacks(
    patience: Optional[int], save_top_k: int
) -> Tuple[callbacks.Callback, List[callbacks.Callback]]:
    """Makes the list of PL callbacks for the trainer.

    Args:
        patience (Optional[int]): patience for stopping training based on
            validation accuracy.
        save_top_k (int): how many of the top-k checkpoints to save.

    Returns:
        Tuple(callbacks.Callback, List[callbacks.Callback]): the checkpoint
            callback for later use, and the list of all callbacks.
    """
    callback = []
    if patience is not None:
        callback.append(
            callbacks.early_stopping.EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=patience,
                verbose=False,
                mode="min",
            )
        )
    ckp_callback = callbacks.ModelCheckpoint(
        save_top_k=save_top_k,
        monitor="val_loss",
        mode="min",
        filename="model-{epoch:02d}-{val_loss:.2f}",
    )
    callback.append(ckp_callback)
    sched_callback = callbacks.LearningRateMonitor(logging_interval="epoch")
    callback.append(sched_callback)
    callback.append(callbacks.TQDMProgressBar())
    return ckp_callback, callback


def make_pl_callbacks(
    patience: Optional[int], save_top_k: int
) -> Tuple[callbacks.Callback, List[callbacks.Callback]]:
    """Makes the list of PL callbacks for the trainer.

    Args:
        patience (Optional[int]): patience for stopping training based on
            validation accuracy.
        save_top_k (int): how many of the top-k checkpoints to save.

    Returns:
        Tuple(callbacks.Callback, List[callbacks.Callback]): the checkpoint
            callback for later use, and the list of all callbacks.
    """
    callback = []
    if patience is not None:
        callback.append(
            callbacks.early_stopping.EarlyStopping(
                monitor="val_accuracy",
                min_delta=0.00,
                patience=patience,
                verbose=False,
                mode="max",
            )
        )
    ckp_callback = callbacks.ModelCheckpoint(
        save_top_k=save_top_k,
        monitor="val_accuracy",
        mode="max",
        filename="model-{epoch:02d}-{val_accuracy:.2f}",
    )
    callback.append(ckp_callback)
    sched_callback = callbacks.LearningRateMonitor(logging_interval="epoch")
    callback.append(sched_callback)
    callback.append(callbacks.TQDMProgressBar())
    return ckp_callback, callback


def pretrain(
    pretraining_data_file: str,
    vocab_dir: str,
    lang: str,
    arch: str,
    dataset_cls: torch.utils.data.Dataset,
    source_col: int,
    target_col: int,
    source_sep: str,
    target_sep: int,
    model_cls: models.base.BaseEncoderDecoder,
    training_args: Dict,
    experiment_name: str,
    output_path: str,
    patience: int,
    save_top_k: int,
    logger: loggers.CSVLogger,
    epochs: int,
    gradient_clip: float,
    eval_every: int,
    batch_size: int,
    device: torch.device,
    gpu: str,
):
    """
    Pretrains the model on a masked language modeling objective to predict masked characters.  Then fine tune on inflection.
    """
    model = model_cls(**training_args).to(device)

    pretrain_set = dataset_cls(
        pretraining_data_file,
        tied_vocabulary=False,
        source_col=source_col,
        target_col=target_col,
        source_sep=source_sep,
        target_sep=target_sep,
    )
    # PL logging.
    logger = [loggers.CSVLogger(output_path, name=f"{experiment_name}-pretraining")]
    pretrain_set.load_index(vocab_dir, lang)
    util.log_info(f"Pretraining source vocabulary: {pretrain_set.source_symbol2i}")
    util.log_info(f"Pretraining target vocabulary: {pretrain_set.target_symbol2i}")
    # ckp_callback is used later for logging the best checkpoint path.
    ckp_callback, pl_callbacks = make_pretraining_pl_callbacks(
        patience=patience, save_top_k=save_top_k
    )
    trainer = pl.Trainer(
        accelerator="gpu" if gpu and torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        max_epochs=epochs,
        gradient_clip_val=gradient_clip,
        check_val_every_n_epoch=eval_every,
        enable_checkpointing=True,
        default_root_dir=output_path,
        callbacks=pl_callbacks,
        log_every_n_steps=int(len(pretrain_set) / batch_size),
        num_sanity_val_steps=0,
    )
    # So we can write indices to it before PL creates it.
    os.makedirs(trainer.loggers[0].log_dir, exist_ok=True)
    pretrain_set.write_index(trainer.loggers[0].log_dir, lang)
    if arch in ["pointer_generator_lstm"]:
        collator_cls = collators.MLMFeaturesCollator
    else:
        collator_cls = collators.MLMCollator

    # min/max vocab ID ignores features, since we assume
    # we have none during pretraining
    valid_vocab = [
        i for i, v in enumerate(pretrain_set.source_i2symbol) if i not in pretrain_set.special_idx and i < pretrain_set.features_idx
    ]

    collator = collator_cls(
        pretrain_set.pad_idx,
        mask_idx=pretrain_set.mask_idx,
        src_vocab_min_idx=min(valid_vocab),
        src_vocab_max_idx=max(valid_vocab),
    )
    train_loader = data.DataLoader(
        pretrain_set,
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=True
    )

    # for src_batch, src_mask, tgt_batch, tgt_mask in train_loader:
    #     prediction_strs = pretrain_set.decode_target(
    #         tgt_batch,
    #         symbols=True,
    #         special=True,
    #     )
    #     source_strs = pretrain_set.decode_source(
    #         src_batch, symbols=True, special=True
    #     )

    #     for s, p in zip(source_strs, prediction_strs):
    #         print(s, p)
    
    trainer.fit(model, train_loader, train_loader)
    util.log_info(f"Loading best pretrained model from {ckp_callback.best_model_path}")
    best_model = model_cls.load_from_checkpoint(
        ckp_callback.best_model_path
    ).to(device)

    # train_loader.do_masking = False
    eval_loader = data.DataLoader(
        pretrain_set,
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=False,
    )
    predict.write_predictions(
        best_model,
        eval_loader,
        os.path.join(trainer.loggers[0].log_dir, "predictions.tsv"),
        arch,
        batch_size,
        source_col,
        target_col,
        source_sep,
        target_sep,
        gpu,
        decode_special=True,
    )
    return best_model
    

@click.command()
@click.option("--lang", required=True)
@click.option("--train-data-path", required=True)
@click.option("--pretraining-data-file")
@click.option("--dev-data-path", required=True)
@click.option("--dev-predictions-path")
@click.option("--source-col", type=int, default=1)
@click.option("--target-col", type=int, default=2)
@click.option(
    "--features-col",
    type=int,
    default=3,
    help="0 indicates no feature column should be used",
)
@click.option("--source-sep", type=str, default="")
@click.option("--target-sep", type=str, default="")
@click.option("--features-sep", type=str, default=";")
@click.option("--tied-vocabulary/--no-tied-vocabulary", default=True)
@click.option("--output-path", required=True)
@click.option("--dataloader-workers", type=int, default=1)
@click.option("--experiment-name", required=True)
@click.option("--seed", type=int, default=time.time_ns())
@click.option("--epochs", type=int, default=20)
@click.option("--pretrain-epochs", type=int)
@click.option("--pretrain-lr", type=float)
@click.option("--pretrain-scheduler", type=str)
@click.option("--pretrain-optimizer", default="adadelta")
@click.option(
    "--pretrain-beta1",
    default=0.9,
    type=float,
    help="beta1 (`--optimizer adam` only)",
)
@click.option(
    "--pretrain-beta2",
    default="0.999",
    type=float,
    help="beta2 (`--optimizer adam` only)",
)
@click.option("--pretrain-warmup-steps", type=int)
@click.option(
    "--arch",
    type=click.Choice(
        [
            "feature_invariant_transformer",
            "lstm",
            "pointer_generator_lstm",
            "transducer",
            "transformer",
        ]
    ),
    required=True,
)
@click.option(
    "--oracle-em-epochs",
    type=int,
    default=0,
    help="Number of EM epochs (`--arch transducer` only)",
)
@click.option(
    "--oracle-factor",
    type=int,
    default=1,
    help="Roll-in schedule parameter (`--arch transducer` only)",
)
@click.option(
    "--sed-params-path",
    type=str,
    default=None,
    help="Path to SED parameters (`transducer` only)",
)
@click.option("--patience", type=int)
@click.option("--learning-rate", type=float, required=True)
@click.option("--label-smoothing", type=float)
@click.option("--gradient-clip", type=float, default=0.0)
@click.option("--batch-size", type=int, default=16)
@click.option("--eval-batch-size", type=int, default=1)
@click.option("--embedding-size", type=int, default=128)
@click.option("--hidden-size", type=int, default=256)
@click.option("--dropout", type=float, default=0.3)
@click.option("--enc-layers", type=int, default=1)
@click.option("--dec-layers", type=int, default=1)
@click.option("--max-seq-len", type=int, default=128)
@click.option("--nhead", type=int, default=4)
@click.option("--dropout", type=float, default=0.1)
@click.option("--optimizer", default="adadelta")
@click.option(
    "--beta1",
    default=0.9,
    type=float,
    help="beta1 (`--optimizer adam` only)",
)
@click.option(
    "--beta2",
    default="0.999",
    type=float,
    help="beta2 (`--optimizer adam` only)",
)
@click.option("--warmup-steps", default=1)
@click.option("--lr-scheduler")
@click.option(
    "--train-from", help="Path to checkpoint to continue training from"
)
@click.option("--bidirectional/--no-bidirectional", type=bool, default=True)
@click.option(
    "--attn/--no-attn",
    type=bool,
    default=True,
    help="Use attention (`--arch lstm` only)",
)
@click.option("--max-decode-len", type=int, default=128)
@click.option("--save-top-k", type=int, default=1)
@click.option("--eval-every", type=int, default=5)
@click.option("--gpu/--no-gpu", default=True)
@click.option("--wandb/--no-wandb", default=False)
def main(
    lang,
    train_data_path,
    pretraining_data_file,
    dev_data_path,
    dev_predictions_path,
    tied_vocabulary,
    source_col,
    target_col,
    features_col,
    source_sep,
    target_sep,
    features_sep,
    output_path,
    dataloader_workers,
    experiment_name,
    seed,
    epochs,
    pretrain_epochs,
    pretrain_lr,
    pretrain_scheduler,
    pretrain_optimizer,
    pretrain_beta1,
    pretrain_beta2,
    pretrain_warmup_steps,
    arch,
    oracle_em_epochs,
    oracle_factor,
    sed_params_path,
    patience,
    learning_rate,
    label_smoothing,
    gradient_clip,
    batch_size,
    eval_batch_size,
    embedding_size,
    hidden_size,
    dropout,
    enc_layers,
    dec_layers,
    max_seq_len,
    nhead,
    optimizer,
    beta1,
    beta2,
    warmup_steps,
    lr_scheduler,
    train_from,
    bidirectional,
    attn,
    max_decode_len,
    save_top_k,
    eval_every,
    gpu,
    wandb,
):
    """Training.

    Args:
        lang (_type_): _description_
        train_data_path (_type_): _description_
        pretraining_data_file (_type_): _description_
        dev_data_path (_type_): _description_
        dev_predictions_path (_type_): _description_
        source_col (_type_): _description_
        target_col (_type_): _description_
        features_col (_type_): _description_
        source_sep (_type_): _description_
        target_sep (_type_): _description_
        features_sep (_type_): _description_
        tied_vocabulary (_type_): _description_
        output_path (_type_): _description_
        dataset (_type_): _description_
        dataloader_workers (_type_): _description_
        experiment_name (_type_): _description_
        seed (_type_): _description_
        epochs (_type_): _description_
        arch (_type_): _description_
        oracle_em_epochs (_type_): _description_
        oracle_factor (_type_): _description_
        sed_params_path (_type_): _description_
        patience (_type_): _description_
        learning_rate (_type_): _description_
        label_smoothing (_type_): _description_
        gradient_clip (_type_): _description_
        batch_size (_type_): _description_
        eval_batch_size (_type_): _description_
        embedding_size (_type_): _description_
        hidden_size (_type_): _description_
        dropout (_type_): _description_
        enc_layers (_type_): _description_
        dec_layers (_type_): _description_
        max_seq_len: (_type_) _description_
        nhead (_type_): _description_
        optimizer (_type_): _description_
        beta1 (_type_): _description_
        beta2 (_type_): _description_
        warmup_steps (_type_): _description_
        scheduler (_type_): _description_
        train_from (_type_): _description_
        bidirectional (_type_): _description_
        attn (_type_): _description_
        max_decode_len (_type_): _description_
        save_top_k (_type_): _description_
        eval_every (_type_): _description_
        gpu (_type_): _description_
        wandb (_type_): _description_
    """
    util.log_info("Arguments:")
    for arg, val in click.get_current_context().params.items():
        util.log_info(f"\t{arg}: {val!r}")
    util.seed(seed)
    device = util.get_device(gpu)
    include_features = features_col != 0
    if target_col == 0:
        raise datasets.Error("target_col must be specified for training")
    dataset_cls = datasets.get_dataset_cls(include_features)
    train_set = dataset_cls(
        train_data_path,
        tied_vocabulary,
        source_col,
        target_col,
        source_sep,
        target_sep,
        features_sep=features_sep,
        features_col=features_col,
    )
    util.log_info(f"Source vocabulary: {train_set.source_symbol2i}")
    util.log_info(f"Target vocabulary: {train_set.target_symbol2i}")
    # PL logging.
    logger = [loggers.CSVLogger(output_path, name=experiment_name)]
    if wandb:
        logger.append(
            loggers.WandbLogger(project=experiment_name, log_model="all")
        )
    # ckp_callback is used later for logging the best checkpoint path.
    ckp_callback, pl_callbacks = make_pl_callbacks(
        patience=patience, save_top_k=save_top_k
    )
    trainer = pl.Trainer(
        accelerator="gpu" if gpu and torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        max_epochs=epochs,
        gradient_clip_val=gradient_clip,
        check_val_every_n_epoch=eval_every,
        enable_checkpointing=True,
        default_root_dir=output_path,
        callbacks=pl_callbacks,
        log_every_n_steps=int(len(train_set) / batch_size),
        num_sanity_val_steps=0,
    )
    # So we can write indices to it before PL creates it.
    os.makedirs(trainer.loggers[0].log_dir, exist_ok=True)
    # TODO: dataloader indexing Dicts should probably be added to model state.
    train_set.write_index(trainer.loggers[0].log_dir, lang)
    collator_cls = collators.get_collator_cls(
        arch, include_features, include_targets=True
    )
    collator = collator_cls(train_set.pad_idx)
    train_loader = data.DataLoader(
        train_set,
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_workers,
    )

    # HARDCODE FOR DEV SET
    # THIS CAN BE ADDED AS AN OPTION BUT IM LAZY
    dev_source_col=1
    dev_target_col=2
    dev_features_col=3
    eval_set = dataset_cls(
        dev_data_path,
        tied_vocabulary,
        dev_source_col,
        dev_target_col,
        source_sep,
        target_sep,
        features_col=dev_features_col,
        features_sep=features_sep,
    )
    eval_set.load_index(trainer.loggers[0].log_dir, lang)
    eval_loader = data.DataLoader(
        eval_set,
        collate_fn=collator,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
    )

    evaluator = evaluators.Evaluator(device=device)
    model_cls = models.get_model_cls(arch, attn, include_features)

    training_args = make_training_args(
        arch=arch,
        train_set=train_set,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        nhead=nhead,
        max_seq_len=max_seq_len,
        optimizer=optimizer,
        beta1=beta1,
        beta2=beta2,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        evaluator=evaluator,
        max_decode_len=max_decode_len,
        dropout=dropout,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        bidirectional=bidirectional,
        label_smoothing=label_smoothing,
        lr_scheduler=lr_scheduler,
        expert=models.expert.get_expert(
            train_set,
            epochs=oracle_em_epochs,
            oracle_factor=oracle_factor,
            sed_params_path=sed_params_path,
        )
        if arch in ["transducer"]
        else None,
    )

    if pretraining_data_file is not None:
        print("Running Pretraining")
        # TODO: Replace all with pretrain args in a function
        training_args["scheduler"] = pretrain_scheduler if pretrain_scheduler is not None else lr_scheduler
        training_args["warmup_steps"] = pretrain_warmup_steps if pretrain_warmup_steps is not None else warmup_steps
        model = pretrain(
            pretraining_data_file,
            trainer.loggers[0].log_dir,
            lang,
            arch,
            dataset_cls,
            source_col,
            target_col,
            source_sep,
            target_sep,
            model_cls,
            training_args,
            experiment_name,
            output_path,
            patience,
            save_top_k,
            logger,
            pretrain_epochs if pretrain_epochs is not None else epochs,
            gradient_clip,
            eval_every,
            batch_size,
            device,
            gpu,
        )

        util.log_info("Finetuning pretrained model on inflection data...")
        # TODO: Change all back to finetuning args in a function
        model.scheduler = lr_scheduler
        model.warmup_steps = warmup_steps
        trainer.fit(model, train_loader, eval_loader, ckpt_path=train_from)
    elif train_from is not None:
        util.log_info(f"Loading model from {train_from}")
        model = model_cls.load_from_checkpoint(train_from).to(device)
        util.log_info("Training...")
        trainer.fit(model, train_loader, eval_loader, ckpt_path=train_from)
    else:
        model = model_cls(**training_args).to(device)
        util.log_info("Training...")
        util.log_info(f"Model: {model_cls.__name__}")
        util.log_info(f"Dataset: {dataset_cls.__name__}")
        util.log_info(f"Collator: {collator_cls.__name__}")
        trainer.fit(model, train_loader, eval_loader)
    util.log_info("Training complete")
    util.log_info(
        f"Best model checkpoint path: {ckp_callback.best_model_path}"
    )
    # Writes development set predictions using the best checkpoint,
    # if a predictions path is specified.
    # TODO: Add beam-width option so we can make predictions with beam search.
    if dev_predictions_path:
        best_model = model_cls.load_from_checkpoint(
            ckp_callback.best_model_path
        ).to(device)
        predict.write_predictions(
            best_model,
            eval_loader,
            dev_predictions_path,
            arch,
            eval_batch_size,
            source_col,
            target_col,
            features_col,
            source_sep,
            target_sep,
            features_sep,
            include_features,
            gpu,
        )


if __name__ == "__main__":
    main()
