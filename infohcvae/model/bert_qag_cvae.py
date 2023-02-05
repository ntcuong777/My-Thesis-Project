import json
import logging
import os
from typing import Dict

import torch
import torch.nn as nn
import torch_optimizer as additional_optim
import torch.optim as optim
import pytorch_lightning as pl
from transformers import BertConfig, BertTokenizer
from infohcvae.model.model_utils import (
    gumbel_softmax, sample_gaussian,
    freeze_neural_model,
)
from infohcvae.model.losses import (
    GaussianKLLoss, CategoricalKLLoss,
    ContinuousKernelMMDLoss, CategoricalMMDLoss,
)
from infohcvae.model.custom.custom_bert_model import CustomBertModel
from infohcvae.model.custom.custom_bert_embeddings import CustomBertEmbedding
from infohcvae.model.custom.discriminator import DiscriminatorNet
from infohcvae.model.encoders import PosteriorEncoder, PriorEncoder
from infohcvae.model.decoders import AnswerDecoder, QuestionDecoder
from infohcvae.model.infomax import JensenShannonInfoMax, AnswerJensenShannonInfoMax
from evaluation.training_eval import eval_vae

__logger__ = logging.Logger(__name__)
__EPS__ = 1e-15

class BertQAGConditionalVae(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.program_args = args

        self.debug = False
        if args.debug:
            self.debug = True

        # self.minibatch_size = args.minibatch_size

        """ Model parameters """
        self.max_c_len = args.max_c_len
        self.max_q_len = args.max_q_len
        self.lr = args.lr
        self.optimizer_algorithm = args.optimizer
        self.loss_log_file = args.loss_log_file
        self.eval_metrics_log_file = args.eval_metrics_log_file

        self.nzqdim = nzqdim = args.nzqdim
        self.nzadim = nzadim = args.nzadim
        self.nza_values = nza_values = args.nza_values

        self.w_bce = args.w_bce
        self.alpha_kl_q = args.alpha_kl_q
        self.alpha_kl_a = args.alpha_kl_a
        self.lambda_wae_q = args.lambda_wae_q
        self.lambda_wae_a = args.lambda_wae_a
        self.lambda_qa_info = args.lambda_qa_info


        """ Initialize model """
        base_model = args.base_model
        config = BertConfig.from_pretrained(base_model)

        self.encoder_bert_nlayers = encoder_bert_nlayers = args.encoder_bert_nlayers
        self.encoder_nlayers = encoder_nlayers = args.encoder_nlayers
        self.encoder_nhidden = encoder_nhidden = args.encoder_nhidden
        self.encoder_dropout = encoder_dropout = args.encoder_dropout
        self.decoder_a_nlayers = decoder_a_nlayers = args.decoder_a_nlayers
        self.decoder_a_nhidden = decoder_a_nhidden = args.decoder_a_nhidden
        self.decoder_a_dropout = decoder_a_dropout = args.decoder_a_dropout
        self.decoder_q_nlayers = decoder_q_nlayers = args.decoder_q_nlayers
        self.decoder_q_nhidden = decoder_q_nhidden = args.decoder_q_nhidden
        self.decoder_q_dropout = decoder_q_dropout = args.decoder_q_dropout
        self.d_model = d_model = config.hidden_size

        embedding = CustomBertEmbedding(base_model)
        freeze_neural_model(embedding)
        bert_model = CustomBertModel(base_model)
        freeze_neural_model(bert_model)

        self.tokenizer = BertTokenizer.from_pretrained(base_model)
        self.vocab_size = vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.sos_id = sos_id = self.tokenizer.cls_token_id
        self.eos_id = eos_id = self.tokenizer.sep_token_id

        """ Define components """
        self.posterior_encoder = PosteriorEncoder(embedding, d_model, encoder_nhidden, encoder_nlayers,
                                                  nzqdim, nzadim, nza_values, dropout=encoder_dropout,
                                                  pad_token_id=self.pad_token_id)

        self.prior_encoder = PriorEncoder(embedding, d_model, encoder_nhidden, encoder_nlayers,
                                          nzqdim, nzadim, nza_values, dropout=encoder_dropout,
                                          pad_token_id=self.pad_token_id)

        self.answer_decoder = AnswerDecoder(bert_model, d_model, nzadim, nza_values, decoder_a_nhidden,
                                            decoder_a_nlayers, dropout=decoder_a_dropout,
                                            pad_token_id=self.pad_token_id)

        self.question_decoder = QuestionDecoder(
            embedding, bert_model, nzqdim, d_model, decoder_q_nhidden, decoder_q_nlayers,
            sos_id, eos_id, vocab_size, dropout=decoder_q_dropout, max_q_len=self.max_q_len,
            pad_token_id=self.pad_token_id)

        self.q_discriminator = DiscriminatorNet(self.encoder_nhidden * 2, nzqdim)
        self.a_discriminator = DiscriminatorNet(self.encoder_nhidden * 2, nzadim * nza_values)

        """ Loss computation """
        self.q_rec_criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.a_rec_criterion = nn.CrossEntropyLoss(ignore_index=self.max_c_len)
        self.gaussian_kl_criterion = GaussianKLLoss()
        self.categorical_kl_criterion = CategoricalKLLoss()

        # Define QA infomax model
        preprocessor = None # nn.Sigmoid()
        qa_discriminator = nn.Bilinear(d_model, decoder_q_nhidden, 1)
        self.qa_infomax = JensenShannonInfoMax(x_preprocessor=preprocessor, y_preprocessor=preprocessor,
                                               discriminator=qa_discriminator)

        self.best_em = self.best_f1 = self.best_bleu = 0.0

        """ Validation """
        with open(args.dev_dir, "r") as f:
            dataset_json = json.load(f)
            self.dev_dataset = dataset_json["data"]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BertQAGConditionalVae")
        parser.add_argument("--base_model", default="bert-base-uncased", type=str)
        parser.add_argument("--encoder_nlayers", type=int, default=1)
        parser.add_argument("--encoder_bert_nlayers", type=int, default=0)
        parser.add_argument("--encoder_nhidden", type=int, default=300)
        parser.add_argument("--encoder_dropout", type=float, default=0.2)
        parser.add_argument("--decoder_a_nlayers", type=int, default=1)
        parser.add_argument("--decoder_a_nhidden", type=int, default=300)
        parser.add_argument("--decoder_a_dropout", type=float, default=0.2)
        parser.add_argument("--decoder_q_nlayers", type=int, default=2)
        parser.add_argument("--decoder_q_nhidden", type=int, default=900)
        parser.add_argument("--decoder_q_dropout", type=float, default=0.3)
        parser.add_argument("--nzqdim", type=int, default=50)
        parser.add_argument("--nzadim", type=int, default=16)
        parser.add_argument("--nza_values", type=int, default=8)
        parser.add_argument("--w_bce", type=float, default=1)
        parser.add_argument("--alpha_kl_q", type=float, default=1)
        parser.add_argument("--alpha_kl_a", type=float, default=1)
        parser.add_argument("--lambda_wae_q", type=float, default=10)
        parser.add_argument("--lambda_wae_a", type=float, default=10)
        parser.add_argument("--lambda_qa_info", type=float, default=1)

        parser.add_argument("--lr", default=1e-3, type=float, help="lr")
        parser.add_argument("--optimizer", default="adamw", choices=["sgd", "adam", "swats", "adamw"], type=str,
                            help="optimizer to use, [\"adam\", \"sgd\", \"swats\", \"adamw\"] are supported")

        return parent_parser

    """ Training-related methods """
    def forward(
            self, c_ids: torch.Tensor = None,
            q_ids: torch.Tensor = None, c_a_mask: torch.Tensor = None,
            run_decoder: bool = True
    ) -> Dict:
        assert self.training, "forward() only use for training mode"

        posterior_zq, posterior_zq_mu, posterior_zq_logvar, \
            posterior_za, posterior_za_logits, posterior_c_h = self.posterior_encoder(
                c_ids, q_ids, c_a_mask, return_hs=True)

        prior_zq, prior_zq_mu, prior_zq_logvar, \
            prior_za, prior_za_logits, prior_c_h = self.prior_encoder(c_ids, return_hs=True)

        start_logits, end_logits = None, None
        lm_logits, mean_embeds = None, (None, None)
        if run_decoder:
            # answer decoding
            start_logits, end_logits =\
                self.answer_decoder(c_ids, posterior_za)
            # question decoding
            lm_logits, mean_embeds = self.question_decoder(
                c_ids, q_ids, c_a_mask, posterior_zq, return_qa_mean_embeds=True)

        out = dict({
            "posterior_za_out": (posterior_za, posterior_za_logits),
            "prior_za_out": (prior_za, prior_za_logits),
            "posterior_zq_out": (posterior_zq, posterior_zq_mu, posterior_zq_logvar),
            "prior_zq_out": (prior_zq, prior_zq_mu, prior_zq_logvar),
            "posterior_c_h": posterior_c_h,
            "prior_c_h": prior_c_h,
            "answer_out": (start_logits, end_logits),
            "question_out": (lm_logits,) + mean_embeds
        })
        return out

    def compute_loss(self, out: Dict, batch: torch.Tensor, optimizer_idx):
        _, q_ids, c_ids, a_mask, start_mask, end_mask, no_q_start_positions, no_q_end_positions = batch

        ##############
        # Optimizers #
        ##############
        # ae_opt, d_opt = self.optimizers()
        # ae_opt.zero_grad()
        # d_opt.zero_grad()

        posterior_za, posterior_za_logits = out["posterior_za_out"]
        prior_za, prior_za_logits = out["prior_za_out"]
        posterior_zq, posterior_zq_mu, posterior_zq_logvar = out["posterior_zq_out"]
        prior_zq, prior_zq_mu, prior_zq_logvar = out["prior_zq_out"]
        posterior_c_h = out["posterior_c_h"]
        prior_c_h = out["prior_c_h"]
        start_logits, end_logits = out["answer_out"]
        q_logits, q_mean_emb, a_mean_emb = out["question_out"]

        if optimizer_idx == 0:
            ###############
            # Optimize AE #
            ###############
            # Compute losses
            # q rec loss
            loss_q_rec = self.w_bce * self.q_rec_criterion(
                q_logits[:, :-1, :].transpose(1, 2).contiguous(), q_ids[:, 1:])

            # a rec loss
            max_c_len = c_ids.size(1)
            no_q_start_positions.clamp_(0, max_c_len)
            no_q_end_positions.clamp_(0, max_c_len)
            loss_start_a_rec = self.a_rec_criterion(start_logits, no_q_start_positions)
            loss_end_a_rec = self.a_rec_criterion(end_logits, no_q_end_positions)
            loss_a_rec = self.w_bce * 0.5 * (loss_start_a_rec + loss_end_a_rec)

            # kl loss
            loss_kl, loss_zq_kl, loss_za_kl = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
            if self.alpha_kl_a > 0.001 or self.alpha_kl_q > 0.001:  # eps = 1e-3
                loss_zq_kl = self.alpha_kl_q * self.gaussian_kl_criterion(
                    posterior_zq_mu, posterior_zq_logvar, prior_zq_mu, prior_zq_logvar)
                loss_za_kl = self.alpha_kl_a * self.categorical_kl_criterion(
                    posterior_za_logits, prior_za_logits)
                loss_kl = loss_zq_kl + loss_za_kl

            # QA info loss
            loss_qa_info = self.lambda_qa_info * self.qa_infomax(q_mean_emb, a_mean_emb)

            total_ae_loss = loss_q_rec + loss_a_rec + loss_kl + loss_qa_info
            current_losses = {
                "total_loss": total_ae_loss,
                "loss_q_rec": loss_q_rec,
                "loss_a_rec": loss_a_rec,
                "loss_kl": loss_kl,
                "loss_qa_info": loss_qa_info,
            }
        else:
            ##########################
            # Optimize Discriminator #
            ##########################
            D_q_real = self.q_discriminator(prior_c_h, prior_zq)
            D_a_real = self.a_discriminator(prior_c_h, prior_za.view(-1, self.nzadim * self.nza_values))

            D_q_fake = self.q_discriminator(posterior_c_h, posterior_zq)
            D_a_fake = self.a_discriminator(posterior_c_h, posterior_za.view(-1, self.nzadim * self.nza_values))

            D_q_loss = self.lambda_wae_q * (-torch.mean(torch.log(D_q_real + __EPS__) + torch.log(1 - D_q_fake + __EPS__)))
            D_a_loss = self.lambda_wae_a * (-torch.mean(torch.log(D_a_real + __EPS__) + torch.log(1 - D_a_fake + __EPS__)))
            D_loss = D_q_loss + D_a_loss

            current_losses = {
                "total_loss": D_loss,
                "loss_a_disc": D_a_loss,
                "loss_q_disc": D_q_loss,
            }
        return current_losses

    def training_step(self, batch, batch_idx, optimizer_idx):
        _, q_ids, c_ids, a_mask, start_mask, end_mask, no_q_start_positions, no_q_end_positions = batch
        out = self.forward(c_ids=c_ids, q_ids=q_ids, c_a_mask=a_mask, run_decoder=(optimizer_idx == 0))

        current_losses = self.compute_loss(out, batch, optimizer_idx)

        if batch_idx % 50 == 0:
            # Log to file
            log_str = ""
            for k, v in current_losses.items():
                log_str += "{:s}={:.4f}; ".format(k, v.item())
            with open(self.loss_log_file, "a") as f:
                f.write(log_str + "\n\n")
        self.log_dict(current_losses, prog_bar=False)
        return current_losses["total_loss"]

    """ Generation-related methods """
    def _generate_answer(self, c_ids, za):
        return_start_logits, return_end_logits = self.answer_decoder(c_ids, za)

        # Get generated answer mask from context ids `c_ids`
        gen_c_a_mask, gen_c_a_start_positions, gen_c_a_end_positions = self.answer_decoder.generate(
            c_ids, start_logits=return_start_logits, end_logits=return_end_logits)
        return gen_c_a_mask, gen_c_a_start_positions, gen_c_a_end_positions, return_start_logits, return_end_logits

    def _generate_question(self, c_ids, c_a_mask, zq):
        return self.question_decoder.generate(c_ids, c_a_mask, zq)

    def generate_qa_from_prior(self, c_ids):
        with torch.no_grad():
            zq, _, _, za, _ = self.prior_encoder(c_ids)

            gen_c_a_mask, gen_c_a_start_positions, gen_c_a_end_positions, _, _ = \
                self._generate_answer(c_ids, za)

            q_ids = self._generate_question(c_ids, gen_c_a_mask, zq)

            return q_ids, gen_c_a_start_positions, gen_c_a_end_positions

    """ Validation-related methods """
    def generate_qa_from_posterior(self, c_ids, q_ids, c_a_mask):
        with torch.no_grad():
            zq, _, _, za, _ = self.posterior_encoder(c_ids, q_ids, c_a_mask)

            """ Generation """
            gen_c_a_mask, gen_c_a_start_positions, gen_c_a_end_positions, start_logits, end_logits = \
                self._generate_answer(c_ids, za)
            question_ids = self._generate_question(c_ids, c_a_mask, zq)

        return question_ids, gen_c_a_start_positions, gen_c_a_end_positions, start_logits, end_logits

    """ Validation-related methods """
    def evaluation(self, val_dataloader):
        all_text_examples = val_dataloader.dataset.all_text_examples
        all_preprocessed_examples = val_dataloader.dataset.all_preprocessed_examples

        posterior_metrics, bleu = eval_vae(
            self.program_args, self, val_dataloader, all_text_examples, all_preprocessed_examples)
        posterior_f1 = posterior_metrics["f1"]
        posterior_em = posterior_metrics["exact_match"]
        bleu = bleu * 100

        if posterior_em > self.best_em:
            self.best_em = posterior_em
            filename = os.path.join(self.program_args.best_model_dir, "model-best_em.ckpt")
            self.trainer.save_checkpoint(filename)
        if posterior_f1 > self.best_f1:
            self.best_f1 = posterior_f1
            filename = os.path.join(self.program_args.best_model_dir, "model-best_f1.ckpt")
            self.trainer.save_checkpoint(filename)
        if bleu > self.best_bleu:
            self.best_bleu = bleu
            filename = os.path.join(self.program_args.best_model_dir, "model-best_bleu.ckpt")
            self.trainer.save_checkpoint(filename)

        with open(os.path.join(self.program_args.model_dir, "metrics.json"), "wt") as f:
            import json
            json.dump({"latest_bleu": bleu, "latest_pos_em": posterior_em, "latest_pos_f1": posterior_f1,
                       "best_bleu": self.best_bleu, "best_em": self.best_em, "best_f1": self.best_f1}, f, indent=4)

        log_str = "{}-th Epochs BLEU : {:02.2f} POS_EM : {:02.2f} POS_F1 : {:02.2f}"
        log_str = log_str.format(self.current_epoch + 1, bleu, posterior_em, posterior_f1)
        with open(self.eval_metrics_log_file, "a") as f:
            f.write(log_str + "\n\n")

    def training_epoch_end(self, outputs):
        if (self.current_epoch + 1) % self.program_args.save_frequency == 0:
            filename = os.path.join(
                self.program_args.save_by_epoch_dir, "model-epoch-{:02d}.ckpt".format(self.current_epoch + 1))
            self.trainer.save_checkpoint(filename)

        if (self.current_epoch + 1) % self.program_args.eval_frequency == 0:
            self.evaluation(self.trainer.val_dataloaders[0])

    def validation_step(self, batch, batch_idx):
        pass # nothing here

    """ Optimizer """
    def configure_optimizers(self):
        params_ae = list(self.posterior_encoder.parameters()) + list(self.answer_decoder.parameters()) \
            + list(self.question_decoder.parameters())
        params_ae = filter(lambda p: p.requires_grad, params_ae)

        params_disc = list(self.posterior_encoder.parameters()) + list(self.prior_encoder.parameters()) \
                              + list(self.q_discriminator.parameters()) + list(self.a_discriminator.parameters())
        params_disc = filter(lambda p: p.requires_grad, params_disc)

        # 1st optimizer is optimizer for AE, 2nd is for discriminator
        disc_lr = 3e-4
        if self.optimizer_algorithm == "sgd":
            optimizers = [optim.SGD(params_ae, lr=self.lr, momentum=0.9, nesterov=False),
                          optim.SGD(params_disc, lr=disc_lr, momentum=0.9, nesterov=False)]
        elif self.optimizer_algorithm == "adam":
            optimizers = [optim.Adam(params_ae, lr=self.lr),
                          optim.Adam(params_disc, lr=disc_lr)]
        elif self.optimizer_algorithm == "adamw":
            optimizers = [optim.AdamW(params_ae, lr=self.lr),
                          optim.AdamW(params_disc, lr=disc_lr)]
        else:
            optimizers = [additional_optim.SWATS(params_ae, lr=self.lr, nesterov=False),
                          additional_optim.SWATS(params_disc, lr=disc_lr, nesterov=False)]
        return optimizers, []
