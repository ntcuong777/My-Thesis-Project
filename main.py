import argparse
import os
import random

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer

from infohcvae.model.bert_qag_cvae import BertQAGConditionalVae
from infohcvae.utils import (
    get_squad_data_loader,
)


def main(run_args):
    tokenizer = AutoTokenizer.from_pretrained(run_args.base_model)

    run_args.device = torch.cuda.current_device()

    train_data, eval_data = None, None
    if run_args.load_saved_dataloader:
        train_data = torch.load(os.path.join(run_args.dataloader_dir, "train_data.pt"))
        eval_data = torch.load(os.path.join(run_args.dataloader_dir, "eval_data.pt"))
        train_dataloader = DataLoader(train_data, args.batch_size, shuffle=True)
        eval_dataloader = DataLoader(eval_data, args.batch_size, shuffle=True)
    else:
        train_dataloader, train_data = get_squad_data_loader(tokenizer, run_args.train_dir,
                                           shuffle=True, is_train_set=True, args=run_args)
        eval_dataloader, eval_data = get_squad_data_loader(tokenizer, run_args.dev_dir,
                                          shuffle=False, is_train_set=False, args=run_args)
        torch.save(train_data, os.path.join(run_args.dataloader_dir, "train_data.pt"))
        torch.save(eval_data, os.path.join(run_args.dataloader_dir, "eval_data.pt"))

    val_em_checkpoint_callback = ModelCheckpoint(dirpath=run_args.best_model_dir, save_top_k=1, monitor="exact_match",
                                                 mode="max", filename="model-{epoch:02d}-best_em-{exact_match:.3f}",
                                                 save_weights_only=True)
    val_f1_checkpoint_callback = ModelCheckpoint(dirpath=run_args.best_model_dir, save_top_k=1, monitor="f1",
                                                 mode="max", filename="model-{epoch:02d}-best_f1-{f1:.3f}",
                                                 save_weights_only=True)
    val_bleu_checkpoint_callback = ModelCheckpoint(dirpath=run_args.best_model_dir, save_top_k=1, monitor="bleu",
                                                   mode="max", filename="model-{epoch:02d}-best_bleu-{f1:.3f}",
                                                   save_weights_only=True)
    train_loss_checkpoint_callback = ModelCheckpoint(dirpath=run_args.save_by_epoch_dir, monitor="total_loss",
                                                     mode="min", filename="model-{epoch:02d}",
                                                     every_n_epochs=run_args.save_frequency, save_weights_only=True,
                                                     save_on_train_epoch_end=True)

    model = BertQAGConditionalVae(run_args)
    callbacks = [val_em_checkpoint_callback, val_f1_checkpoint_callback, val_bleu_checkpoint_callback,
                 val_bleu_checkpoint_callback, train_loss_checkpoint_callback]
    full_trainer = Trainer.from_argparse_args(run_args, callbacks=callbacks)
    ckpt_path = args.checkpoint_file
    full_trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=eval_dataloader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=3163, type=int)
    parser.add_argument('--train_dir', default='./data/squad/train-v1.1.json')
    parser.add_argument('--dev_dir', default='./data/squad/my_dev.json')

    parser.add_argument("--max_c_len", default=384, type=int, help="max context length")
    parser.add_argument("--max_q_len", default=64, type=int, help="max query length")
    parser.add_argument("--load_saved_dataloader", dest="load_saved_dataloader",
                        action="store_true", default=False)

    parser.add_argument("--model_dir", default="./save/vae-checkpoint", type=str)
    parser.add_argument("--bart_decoder_finetune_epochs", default=3, type=int)
    parser.add_argument("--dataloader_dir", default="./save/dataloader", type=str)
    parser.add_argument("--checkpoint_file", default=None, type=str,
                        help="Path to the .pt file, None if checkpoint should not be loaded")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--minibatch_size", default=16, type=int, help="mini-batch size")
    parser.add_argument("--loss_log_file", default="./train_loss_info.log", type=str)
    parser.add_argument("--eval_metrics_log_file", default="./metrics_log.log", type=str)
    parser.add_argument("--save_frequency", default=5, type=int, help="save frequency by epoch")

    # Add model-specific args
    parser = BertQAGConditionalVae.add_model_specific_args(parser)

    # add all the available pytorch lightning trainer options to argparse
    # ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    if args.fast_dev_run:
        args.debug = True
    else:
        args.debug = False

    # set model dir
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    args.model_dir = os.path.abspath(model_dir)

    save_by_epoch_dir = os.path.join(args.model_dir, "per_epoch")
    best_model_dir = os.path.join(args.model_dir, "best_models")
    os.makedirs(save_by_epoch_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    args.save_by_epoch_dir = save_by_epoch_dir
    args.best_model_dir = best_model_dir

    # set dataloader dir
    if not args.load_saved_dataloader:
        dataloader_dir = args.dataloader_dir
        os.makedirs(dataloader_dir, exist_ok=True)
        args.dataloader_dir = os.path.abspath(dataloader_dir)

    open(args.loss_log_file, "w")  # empty loss log file if existed
    open(args.eval_metrics_log_file, "w")  # empty loss log file if existed

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    # Main processing
    main(args)