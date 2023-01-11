import collections

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import T5Tokenizer, T5Config
from infohcvae.model.encoders import PosteriorQAEncoder
from infohcvae.model.decoders import QuestionDecoder, AnswerDecoder
from infohcvae.model.losses import (
    ContinuousKernelMMDLoss,
    GumbelMMDLoss,
    VaeGaussianKLLoss,
    VaeGumbelKLLoss,
)
from infohcvae.model.model_utils import gumbel_latent_var_sampling
from infohcvae.model.infomax.jensen_shannon_infomax import JensenShannonInfoMax
from infohcvae.model.custom import CustomT5Encoder


class DiscreteVAE(pl.LightningModule):
    def __init__(self, args):
        super(DiscreteVAE, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(args.huggingface_model)
        padding_idx = self.tokenizer.pad_token_id
        sos_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.sep_token_id

        self.lr = args.lr

        base_model = args.huggingface_model

        enc_finetune_nlayers = args.enc_finetune_nlayers
        dec_a_finetune_nlayers = args.dec_a_finetune_nlayers
        dec_q_finetune_nlayers = args.dec_q_finetune_nlayers
        self.nzqdim = nzqdim = args.nzqdim
        self.nzadim = nzadim = args.nzadim
        self.nza_values = nza_values = args.nza_values

        self.w_bce = args.w_bce
        self.alpha_kl_q = args.alpha_kl_q
        self.alpha_kl_a = args.alpha_kl_a
        self.lambda_mmd_q = args.lambda_mmd_q
        self.lambda_mmd_a = args.lambda_mmd_a
        self.lambda_qa_info = args.lambda_qa_info

        max_q_len = args.max_q_len

        base_encoder = CustomT5Encoder(base_model=base_model, num_enc_finetune_layers=enc_finetune_nlayers)

        self.posterior_encoder = PosteriorQAEncoder(padding_idx, base_encoder, nzqdim, nzadim, nza_values,
                                                    base_model=base_model,
                                                    pooling_strategy=args.pooling_strategy,
                                                    num_enc_finetune_layers=enc_finetune_nlayers)

        self.answer_decoder = AnswerDecoder(padding_idx, nzadim, nza_values,
                                            base_model=base_model, n_dec_finetune_layers=dec_a_finetune_nlayers)

        self.question_decoder = QuestionDecoder(sos_id, eos_id, padding_idx, base_encoder, nzqdim,
                                                num_dec_finetune_layers=dec_q_finetune_nlayers,
                                                base_model=base_model, max_q_len=max_q_len)

        self.q_rec_criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
        self.a_rec_criterion = nn.CrossEntropyLoss(ignore_index=args.max_c_len)
        self.gaussian_kl_criterion = VaeGaussianKLLoss()
        self.categorical_kl_criterion = VaeGumbelKLLoss(categorical_dim=nza_values)

        self.cont_mmd_criterion = ContinuousKernelMMDLoss()
        self.gumbel_mmd_criterion = GumbelMMDLoss()

        config = T5Config.from_pretrained(base_model)

        # Define QA infomax model
        preprocessor = nn.Sigmoid()
        qa_discriminator = nn.Bilinear(config.d_model, config.d_model, 1)
        self.qa_infomax = JensenShannonInfoMax(x_preprocessor=preprocessor, y_preprocessor=preprocessor,
                                               discriminator=qa_discriminator)

    def forward(self, c_ids, q_ids, a_mask, start_positions, end_positions):
        za_logits, za, zq_mu, zq_logvar, zq \
            = self.posterior_encoder(c_ids, q_ids, a_mask, return_distribution_parameters=True)

        # answer decoding
        start_logits, end_logits = self.answer_decoder(c_ids, za)
        # question decoding
        q_logits, (q_mean_emb, a_mean_emb) = self.question_decoder(c_ids, q_ids, a_mask, zq,
                                                                   return_qa_mean_embeds=True)

        # Compute losses
        # q rec loss
        loss_q_rec = self.q_rec_criterion(q_logits[:, :-1, :].transpose(1, 2).contiguous(),
                                          q_ids[:, 1:])

        # a rec loss
        max_c_len = c_ids.size(1)
        start_positions.clamp_(0, max_c_len)
        end_positions.clamp_(0, max_c_len)
        loss_start_a_rec = self.a_rec_criterion(
            start_logits, start_positions)
        loss_end_a_rec = self.a_rec_criterion(end_logits, end_positions)
        loss_a_rec = loss_start_a_rec + loss_end_a_rec

        # kl loss
        loss_zq_kl = (1. - self.alpha_kl_q) * self.gaussian_kl_criterion(zq_mu, zq_logvar)
        loss_za_kl = (1. - self.alpha_kl_a) * self.categorical_kl_criterion(za_logits)

        loss_zq_mmd = (self.alpha_kl_q + self.lambda_mmd_q - 1.) * self.cont_mmd_criterion(zq)
        loss_za_mmd = (self.alpha_kl_a + self.lambda_mmd_a - 1.) * self.cont_mmd_criterion(za)
        loss_mmd = loss_zq_mmd + loss_za_mmd

        # QA info loss
        loss_qa_info = self.qa_infomax(q_mean_emb, a_mean_emb)

        loss_kl = loss_zq_kl + loss_za_kl
        loss_qa_info = self.lambda_qa_info * loss_qa_info
        total_loss = self.w_bce * (loss_q_rec + loss_a_rec) + \
                     loss_kl + loss_qa_info + loss_mmd

        current_losses = {
            "total_loss": total_loss,
            "loss_q_rec": loss_q_rec,
            "loss_a_rec": loss_a_rec,
            "loss_kl": loss_kl,
            "loss_zq_kl": loss_zq_kl,
            "loss_za_kl": loss_za_kl,
            "loss_mmd": loss_mmd,
            "loss_zq_mmd": loss_zq_mmd,
            "loss_za_mmd": loss_za_mmd,
            "loss_qa_info": loss_qa_info,
        }

        return current_losses

    def generate_qa(self, c_ids, zq=None, za=None):
        a_mask, start_positions, end_positions = self.answer_decoder.generate(c_ids, za)
        q_ids = self.question_decoder.generate(c_ids, a_mask, zq)
        return q_ids, start_positions, end_positions

    def generate_qa_from_posterior(self, c_ids, q_ids, a_mask):
        with torch.no_grad():
            zq, za = self.posterior_encoder(c_ids, q_ids, a_mask)
            q_ids, start_positions, end_positions = self.generate_qa(
                c_ids, zq=zq, za=za)
        return q_ids, start_positions, end_positions, zq

    def return_answer_logits(self, c_ids, za=None):
        start_logits, end_logits = self.answer_decoder(c_ids, za)
        return start_logits, end_logits

    def generate_answer_logits_from_posterior(self, c_ids, q_ids, a_mask):
        with torch.no_grad():
            za, zq = self.posterior_encoder(c_ids, q_ids, a_mask)
            start_logits, end_logits = self.return_answer_logits(c_ids, za=za)
        return start_logits, end_logits

    def generate_qa_from_prior(self, c_ids):
        with torch.no_grad():
            zq = torch.randn(c_ids.size(0), self.nzqdim).to(c_ids.device)
            za = gumbel_latent_var_sampling(c_ids.size(0), self.nzadim, self.nza_values, c_ids.device)
            q_ids, start_positions, end_positions = self.generate_qa(c_ids, zq=zq, za=za)
        return q_ids, start_positions, end_positions, zq
