import collections
import json
import logging
import itertools
from copy import deepcopy
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as additional_optim
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
from torch_scatter import scatter_max
from transformers import BertModel, BertConfig, BertTokenizer
from transformers.models.bert.modeling_bert import BertPooler
from infohcvae.model.model_utils import (
    return_attention_mask, cal_attn,
    gumbel_softmax, sample_gaussian,
    gumbel_latent_var_sampling,
)
from infohcvae.model.losses import (
    VaeGaussianKLLoss, VaeGumbelKLLoss,
    ContinuousKernelMMDLoss, GumbelMMDLoss,
)
from infohcvae.model.custom.custom_torch_dataset import CustomDataset
from infohcvae.model.infomax.jensen_shannon_infomax import JensenShannonInfoMax
from infohcvae.squad_utils import evaluate, extract_predictions_to_dict
from evaluation.qgevalcap.eval import eval_qg

__logger__ = logging.Logger(__name__)


class BertQAGConditionalVae(pl.LightningModule):
    def __init__(self, args, dropout=0.3):
        super().__init__()
        self.save_hyperparameters()

        self.debug = False
        if args.debug:
            self.debug = True

        """ Model parameters """
        self.max_c_len = args.max_c_len
        self.max_q_len = args.max_q_len
        self.lr = args.lr
        self.optimizer_algorithm = args.optimizer
        self.loss_log_file = args.loss_log_file
        self.eval_metrics_log_file = args.eval_metrics_log_file

        self.num_finetune_enc_layers = args.num_finetune_enc_layers
        self.num_dec_layers = args.num_dec_layers
        self.bart_decoder_finetune_epochs = args.bart_decoder_finetune_epochs

        self.nzqdim = nzqdim = args.nzqdim
        self.nzadim = nzadim = args.nzadim
        self.nza_values = nza_values = args.nza_values

        self.alpha_kl_q = args.alpha_kl_q
        self.alpha_kl_a = args.alpha_kl_a
        self.lambda_mmd_q = args.lambda_mmd_q
        self.lambda_mmd_a = args.lambda_mmd_a
        self.lambda_qa_info = args.lambda_qa_info

        self.pooling_strategy = args.pooling_strategy

        """ Initialize model """
        base_model = args.base_model
        config = BertConfig.from_pretrained(base_model)
        bert_model = BertModel.from_pretrained(base_model)

        self.encoder_nlayers = config.num_hidden_layers
        self.decoder_nlayers = decoder_nlayers = min(config.num_hidden_layers, self.num_dec_layers)
        self.decoder_nheads = decoder_nheads = config.num_attention_heads
        self.d_model = d_model = config.hidden_size

        self.tokenizer = BertTokenizer.from_pretrained(base_model, add_pooling_layer=False)
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        self.sos_id = self.tokenizer.cls_token_id
        self.eos_id = self.tokenizer.sep_token_id

        # Initialize posterior encoder
        self.posterior_encoder = bert_model

        # Pooling layer for computing the latent variables
        self.first_token_pooler = BertPooler(config) if self.pooling_strategy == "first" else None

        # Freeze embedding layer
        enc_embedding_layer = self.posterior_encoder.get_input_embeddings()
        for params in self.posterior_encoder.get_input_embeddings().parameters():
            params.requires_grad = False
        # freeze encoder layers, except `num_finetune_enc_layers` top layers
        for i in range(self.encoder_nlayers - self.num_finetune_enc_layers):
            for param in self.posterior_encoder.encoder.layer[i].parameters():
                param.requires_grad = False

        # self.use_neural_prior = True if args.use_neural_prior == 1 else False
        # if self.use_neural_prior:
        #     # Initialize prior encoder
        #     self.prior_encoder = deepcopy(self.posterior_encoder)

        # Initialize answer & question decoder
        decoder = BertModel.from_pretrained(base_model, is_decoder=True, add_cross_attention=True,
                                            add_pooling_layer=False)
        # freeze decoder embedding layers
        for params in decoder.get_input_embeddings().parameters():
            params.requires_grad = False
        # Only use the first `num_dec_layers` for answer & question decoding
        decoder.encoder.layer = decoder.encoder.layer[:self.num_dec_layers]

        self.answer_decoder = decoder
        self.question_decoder = deepcopy(decoder)

        """ Encoder properties """
        self.embed_size_per_head = d_model // decoder_nheads

        self.question_attention = nn.Linear(d_model, d_model)
        self.context_attention = nn.Linear(d_model, d_model)

        self.za_linear = nn.Linear(d_model, nzadim * nza_values, bias=False)
        self.zq_mu_linear = nn.Linear(4*d_model, nzqdim, bias=False)
        self.zq_logvar_linear = nn.Linear(4*d_model, nzqdim, bias=False)

        """ Answer decoder properties """
        self.za_memory_projection = nn.Linear(
            nzadim * nza_values,
            decoder_nlayers * decoder_nheads * self.embed_size_per_head,
            bias=False,
        )
        self.start_linear = nn.Linear(d_model, 1)
        self.end_linear = nn.Linear(d_model, 1)

        """ Question decoder properties """
        self.zq_memory_projection = nn.Linear(
            nzqdim,
            decoder_nlayers * decoder_nheads * self.embed_size_per_head,
            bias=False,
        )

        self.question_linear = nn.Linear(d_model, d_model)
        self.concat_linear = nn.Sequential(nn.Linear(2 * d_model, 2 * d_model),
                                           nn.Dropout(dropout),
                                           nn.Linear(2 * d_model, 2 * d_model))

        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)

        # fix output word matrix
        self.lm_head.weight = enc_embedding_layer.weight
        for param in self.lm_head.parameters():
            param.requires_grad = False

        """ Loss computation """
        self.q_rec_criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.a_rec_criterion = nn.CrossEntropyLoss(ignore_index=self.max_c_len)
        self.gaussian_kl_criterion = VaeGaussianKLLoss()
        self.categorical_kl_criterion = VaeGumbelKLLoss(categorical_dim=nza_values)

        self.cont_mmd_criterion = ContinuousKernelMMDLoss()
        self.gumbel_mmd_criterion = GumbelMMDLoss()

        # Define QA infomax model
        preprocessor = nn.Sigmoid()
        qa_discriminator = nn.Bilinear(d_model, d_model, 1)
        self.qa_infomax = JensenShannonInfoMax(x_preprocessor=preprocessor, y_preprocessor=preprocessor,
                                               discriminator=qa_discriminator)

        """ Validation """
        with open(args.dev_dir, "r") as f:
            dataset_json = json.load(f)
            self.dev_dataset = dataset_json["data"]
        self.example_idx = -1 # to keep track of the current index of example

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BertQAGConditionalVae")
        parser.add_argument("--base_model", default='bert-base-uncased', type=str)
        parser.add_argument('--num_finetune_enc_layers', type=int, default=1)
        parser.add_argument('--num_dec_layers', type=int, default=3)
        # parser.add_argument('--use_neural_prior', type=int, default=0, choices=[0, 1])
        parser.add_argument('--nzqdim', type=int, default=64)
        parser.add_argument('--nzadim', type=int, default=20)
        parser.add_argument('--nza_values', type=int, default=10)
        parser.add_argument('--pooling_strategy', type=str, default="first", choices=["max", "mean", "first"])
        parser.add_argument('--alpha_kl_q', type=float, default=1)
        parser.add_argument('--alpha_kl_a', type=float, default=1)
        parser.add_argument('--lambda_mmd_q', type=float, default=2300)
        parser.add_argument('--lambda_mmd_a', type=float, default=137)
        parser.add_argument('--lambda_qa_info', type=float, default=1)

        parser.add_argument("--lr", default=1e-3, type=float, help="lr")
        parser.add_argument("--optimizer", default="manual", choices=["sgd", "adam", "swats", "adamw"], type=str,
                            help="optimizer to use, [\"adam\", \"sgd\", \"swats\", \"adamw\"] are supported")

        return parent_parser

    def pool(self, last_hidden_states):
        # last_hidden_states shape = (batch_size, seq_length, hidden_size)
        # pooling over `seq_length` dim
        if self.pooling_strategy == "mean":
            return last_hidden_states.mean(dim=1)
        elif self.pooling_strategy == "max":
            return torch.max(last_hidden_states, dim=1)[0]  # Pool from last layer
        elif self.pooling_strategy == "first":
            return self.first_token_pooler(last_hidden_states)
        else:
            raise Exception("Wrong pooling strategy!")

    def calculate_zq_latent(self, pooled):
        zq_mu, zq_logvar = self.zq_mu_linear(pooled), self.zq_logvar_linear(pooled)
        zq = sample_gaussian(zq_mu, zq_logvar)
        return zq, zq_mu, zq_logvar

    def build_zq_past(self, zq):
        projection = self.zq_memory_projection(zq)
        cross_attn = projection.reshape(
            self.decoder_nlayers,
            projection.shape[0],
            self.decoder_nheads,
            1,
            self.embed_size_per_head,
        )
        past_key_values = tuple((ca, ca) for ca in cross_attn)
        return past_key_values

    def calculate_za_latent(self, pooled, hard=True):
        za_logits = self.za_linear(pooled).view(-1, self.nzadim, self.nza_values)
        za = gumbel_softmax(za_logits, hard=hard)
        return za, za_logits

    def build_za_past(self, za):
        projection = self.za_memory_projection(za.view(-1, self.nzadim * self.nza_values))
        cross_attn = projection.reshape(
            self.decoder_nlayers,
            projection.shape[0],
            self.decoder_nheads,
            1,
            self.embed_size_per_head,
        )
        past_key_values = tuple((ca, ca) for ca in cross_attn)
        return past_key_values

    def _encode_input_tokens(self, encoder, input_ids, input_mask, subspan_mask=None):
        encoder_outputs = encoder(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=subspan_mask
        )
        return encoder_outputs[0]

    def _encode_latent_posteriors(self, q_ids, q_mask, c_ids, c_mask, c_a_mask):
        """ START: Answer encoding to get latent `za` """
        # context enc
        c_a_hidden_states = self._encode_input_tokens(
            encoder=self.posterior_encoder, input_ids=c_ids, input_mask=c_mask, subspan_mask=c_a_mask)

        # sample `za`
        za, za_logits = self.calculate_za_latent(self.pool(c_a_hidden_states), hard=True)
        """ END: Answer encoding to get latent `za` """

        """ START: Question encoding to get latent `zq` """
        # question-context & answer enc
        q_hidden_states = self._encode_input_tokens(
            encoder=self.posterior_encoder, input_ids=q_ids, input_mask=q_mask)
        q_pooled = self.pool(q_hidden_states)

        # attetion q, c
        mask = c_mask.unsqueeze(1)
        c_attned_by_q, _ = cal_attn(self.question_attention(q_pooled).unsqueeze(1),
                                    c_a_hidden_states, mask)
        c_attned_by_q = c_attned_by_q.squeeze(1)

        # attetion c, q
        c_a_pooled = self.pool(c_a_hidden_states)
        mask = q_mask.unsqueeze(1)
        q_attned_by_c, _ = cal_attn(self.context_attention(c_a_pooled).unsqueeze(1),
                                    q_hidden_states, mask)
        q_attned_by_c = q_attned_by_c.squeeze(1)

        h = torch.cat([q_pooled, q_attned_by_c, c_a_pooled, c_attned_by_q], dim=-1)
        # sample `zq`
        zq, zq_mu, zq_logvar = self.calculate_zq_latent(h)
        """ END: Question encoding to get latent `zq` """
        return zq, zq_mu, zq_logvar, za, za_logits, c_a_hidden_states

    def _decode_answer(self, c_ids, c_mask, za):
        # Initialize `past_key_values` with `za` for question generation
        za_past_key_values = self.build_za_past(za)

        # extend the input attention mask so that the init state from `za` is attended during self-attention
        past_attended_c_mask = torch.cat([torch.ones(c_ids.size(0), 1).to(c_ids.device), c_mask], dim=-1)
        answer_decoder_ouputs = self.answer_decoder(
            input_ids=c_ids,
            attention_mask=past_attended_c_mask,
            past_key_values=za_past_key_values
        )
        answer_out_hidden_states = answer_decoder_ouputs[0]

        start_logits = self.start_linear(answer_out_hidden_states).squeeze(-1)
        end_logits = self.end_linear(answer_out_hidden_states).squeeze(-1)

        start_end_mask = (c_mask == 0)
        start_logits = start_logits.masked_fill(start_end_mask, -10000.0)
        end_logits = end_logits.masked_fill(start_end_mask, -10000.0)
        return start_logits, end_logits

    def _decode_question(self, q_ids, q_mask, c_ids, c_a_hidden_states, c_mask, zq,
                         past_key_values=None, use_cache=None):
        def get_question_logits_from_out_hidden_states(q_out_hidden_states):
            N, max_q_len = q_ids.size()
            # gen logits
            gen_logits = self.lm_head(q_out_hidden_states)

            # copy logits
            # context-question attention
            mask = torch.matmul(q_mask.unsqueeze(2), c_mask.unsqueeze(1))
            _, attn_logits = cal_attn(self.question_linear(q_out_hidden_states), c_a_hidden_states, mask)

            bq = N * max_q_len
            context_ids = c_ids.unsqueeze(1).repeat(
                1, max_q_len, 1).view(bq, -1).contiguous()
            attn_logits = attn_logits.view(bq, -1).contiguous()
            copy_logits = torch.zeros(bq, self.vocab_size).to(context_ids.device) - 10000.0
            copy_logits, _ = scatter_max(attn_logits, context_ids, out=copy_logits)
            copy_logits = copy_logits.masked_fill(copy_logits == -10000.0, 0)
            copy_logits = copy_logits.view(N, max_q_len, -1).contiguous()

            total_logits = gen_logits + copy_logits
            return total_logits

        if past_key_values is None:
            past_key_values = self.build_zq_past(zq)
            past_key_values_tmp = []
            for (k, v) in past_key_values:
                k = k.expand(-1, -1, c_ids.size(1), -1)
                v = v.expand(-1, -1, c_ids.size(1), -1)
                past_key_values_tmp.append((k, v))
            past_key_values = past_key_values_tmp

        # extend the input attention mask so that the init state from `za` is attended during self-attention
        past_key_values_length = past_key_values[0][0].shape[2]
        past_attended_q_mask = \
            torch.cat([torch.ones(q_ids.size(0), past_key_values_length).to(q_ids.device), q_mask], dim=-1)
        question_decoder_outputs = self.question_decoder(
            input_ids=q_ids,
            attention_mask=past_attended_q_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=c_a_hidden_states,
            encoder_attention_mask=c_mask,
            use_cache=use_cache
        )
        question_out_hidden_states = question_decoder_outputs[0]

        present_key_values = None
        if use_cache is not None and use_cache:
            present_key_values = question_decoder_outputs[1]

        # Generate question id
        lm_logits = get_question_logits_from_out_hidden_states(question_out_hidden_states)
        return lm_logits, question_out_hidden_states, present_key_values

    def forward(
            self, c_ids: torch.Tensor = None,
            q_ids: torch.Tensor = None, c_a_mask: torch.Tensor = None,
            return_qa_mean_embeds: Optional[bool] = None,
    ) -> Dict:
        assert self.training, "forward() only use for training mode"

        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        q_mask = return_attention_mask(q_ids, self.pad_token_id)

        """ ENCODING PART or LATENT CODE GENERATION PART """
        zq, zq_mu, zq_logvar, za, za_logits, c_a_hidden_states = self._encode_latent_posteriors(
            q_ids, q_mask, c_ids, c_mask, c_a_mask)

        """ ANSWER DECODING PART """
        start_logits, end_logits = self._decode_answer(c_ids, c_mask, za)

        """ QUESTION DECODING PART """
        lm_logits, question_out_hidden_states, _ = self._decode_question(
            q_ids, q_mask, c_ids, c_a_hidden_states, c_mask, zq)

        out_mean_qa = (None, None)
        if return_qa_mean_embeds is not None and return_qa_mean_embeds:
            # mutual information btw averged answer and question representations
            a_emb = c_a_hidden_states * c_a_mask.float().unsqueeze(2)
            a_mean_emb = torch.sum(a_emb, dim=1) / c_a_mask.sum(1).unsqueeze(1).float()

            q_emb = question_out_hidden_states * q_mask.unsqueeze(2)
            q_mean_emb = torch.sum(q_emb, dim=1) / q_mask.sum(dim=1, keepdim=True).float()

            out_mean_qa = (q_mean_emb, a_mean_emb)

        out = dict({
            "za_out": (za, za_logits),
            "zq_out": (zq, zq_mu, zq_logvar),
            "answer_out": (start_logits, end_logits),
            "question_out": (lm_logits,) + out_mean_qa
        })
        return out

    def training_step(self, batch, batch_idx):
        _, q_ids, c_ids, a_mask, _, no_q_start_positions, no_q_end_positions = batch
        out = self.forward(c_ids=c_ids, q_ids=q_ids, c_a_mask=a_mask, return_qa_mean_embeds=True)

        za, za_logits = out["za_out"]
        zq, zq_mu, zq_logvar = out["zq_out"]
        start_logits, end_logits = out["answer_out"]
        q_logits, q_mean_emb, a_mean_emb = out["question_out"]

        # Compute losses
        # q rec loss
        loss_q_rec = self.q_rec_criterion(q_logits[:, :-1, :].transpose(1, 2).contiguous(),
                                          q_ids[:, 1:])

        # a rec loss
        max_c_len = c_ids.size(1)
        no_q_start_positions.clamp_(0, max_c_len)
        no_q_end_positions.clamp_(0, max_c_len)
        loss_start_a_rec = self.a_rec_criterion(start_logits, no_q_start_positions)
        loss_end_a_rec = self.a_rec_criterion(end_logits, no_q_end_positions)
        loss_a_rec = loss_start_a_rec + loss_end_a_rec

        # kl loss
        loss_kl, loss_zq_kl, loss_za_kl = 0.0, 0.0, 0.0
        if self.alpha_kl_a < 1. or self.alpha_kl_q < 1.:
            loss_zq_kl = (1. - self.alpha_kl_q) * self.gaussian_kl_criterion(zq_mu, zq_logvar)
            loss_za_kl = (1. - self.alpha_kl_a) * self.categorical_kl_criterion(za_logits)
            loss_kl = loss_zq_kl + loss_za_kl

        loss_zq_mmd = (self.alpha_kl_q + self.lambda_mmd_q - 1.) * self.cont_mmd_criterion(zq)
        loss_za_mmd = (self.alpha_kl_a + self.lambda_mmd_a - 1.) * self.gumbel_mmd_criterion(za)
        loss_mmd = loss_zq_mmd + loss_za_mmd

        # QA info loss
        loss_qa_info = self.lambda_qa_info * self.qa_infomax(q_mean_emb, a_mean_emb)

        total_loss = loss_q_rec + loss_a_rec + loss_kl + loss_qa_info + loss_mmd

        current_losses = {
            "loss_q_rec": loss_q_rec,
            "loss_a_rec": loss_a_rec,
            "loss_mmd": loss_mmd,
            "loss_zq_mmd": loss_zq_mmd,
            "loss_za_mmd": loss_za_mmd,
            "loss_kl": loss_kl,
            "loss_qa_info": loss_qa_info,
        }

        # Log to file
        log_str = ""
        for k, v in current_losses.items():
            log_str += "{:s}={:.4f}; ".format(k, v)
        with open(self.loss_log_file, "a") as f:
            f.write(log_str + "\n\n")
        # self.log_dict(current_losses, prog_bar=False)

        return total_loss

    def _generate_answer(self, c_ids, c_mask, za):
        def generate_answer_mask_from_context(start_logits, end_logits):
            batch_size, max_c_len = c_ids.size()

            mask = torch.matmul(c_mask.unsqueeze(2).float(),
                                c_mask.unsqueeze(1).float())
            mask = torch.triu(mask) == 0
            score = (F.log_softmax(start_logits, dim=1).unsqueeze(2)
                     + F.log_softmax(end_logits, dim=1).unsqueeze(1))
            score = score.masked_fill(mask, -10000.0)
            score, start_positions = score.max(dim=1)
            score, end_positions = score.max(dim=1)
            start_positions = torch.gather(start_positions, 1, end_positions.view(-1, 1)).squeeze(1)

            idxes = torch.arange(0, max_c_len, out=torch.LongTensor(max_c_len))
            idxes = idxes.unsqueeze(0).to(
                start_logits.device).repeat(batch_size, 1)

            start_positions = start_positions.unsqueeze(1)
            start_mask = (idxes >= start_positions).long()
            end_positions = end_positions.unsqueeze(1)
            end_mask = (idxes <= end_positions).long()
            a_mask = start_mask + end_mask - 1

            return a_mask, start_positions.squeeze(1), end_positions.squeeze(1)

        return_start_logits, return_end_logits = self._decode_answer(c_ids, c_mask, za)

        # Get generated answer mask from context ids `c_ids`
        gen_c_a_mask, gen_c_a_start_positions, gen_c_a_end_positions = generate_answer_mask_from_context(
            return_start_logits, return_end_logits)
        return gen_c_a_mask, gen_c_a_start_positions, gen_c_a_end_positions, return_start_logits, return_end_logits

    def _generate_question(self, c_ids, c_mask, a_mask, zq):
        """ Greedy decoding """

        def postprocess(q_ids):
            eos_mask = (q_ids == self.eos_id).float()
            no_eos_idx_sum = (eos_mask.sum(dim=1) == 0).long() * \
                             (self.max_q_len - 1)
            eos_mask = eos_mask.cpu().numpy()
            q_lengths = np.argmax(eos_mask, axis=1) + 1
            q_lengths = torch.tensor(q_lengths).to(
                q_ids.device).long() + no_eos_idx_sum
            batch_size, max_len = q_ids.size()
            idxes = torch.arange(0, max_len).to(q_ids.device)
            idxes = idxes.unsqueeze(0).repeat(batch_size, 1)
            q_mask = (idxes < q_lengths.unsqueeze(1))
            q_ids = q_ids.long() * q_mask.long()
            return q_ids

        batch_size = c_ids.size(0)

        # Init state, only BOS token is available for each question
        q_ids = torch.LongTensor([self.sos_id] * batch_size).unsqueeze(1).to(c_ids.device)
        q_mask = return_attention_mask(q_ids, self.pad_token_id)

        past_key_values = None

        c_a_hidden_states = self._encode_input_tokens(
            encoder=self.posterior_encoder, input_ids=c_ids, input_mask=c_mask, subspan_mask=a_mask)

        # unroll
        all_q_ids = list()
        all_q_ids.append(q_ids)
        for _ in range(self.max_q_len - 1):
            lm_logits, _, past_key_values = self._decode_question(
                q_ids, q_mask, c_ids, c_a_hidden_states, c_mask, zq,
                past_key_values=past_key_values, use_cache=True)

            q_ids = torch.argmax(lm_logits, dim=2)
            all_q_ids.append(q_ids)
            q_mask = return_attention_mask(q_ids, self.pad_token_id)

        q_ids = torch.cat(all_q_ids, dim=1)
        q_ids = postprocess(q_ids)

        return q_ids

    def generate_qa_from_prior(self, c_ids):
        with torch.no_grad():
            c_mask = return_attention_mask(c_ids, self.pad_token_id)

            zq = torch.randn(c_ids.size(0), self.nzqdim).to(c_ids.device)
            za = gumbel_latent_var_sampling(c_ids.size(0), self.nzadim, self.nza_values, c_ids.device)

            gen_c_a_mask, gen_c_a_start_positions, gen_c_a_end_positions, _, _ = \
                self._generate_answer(c_ids, c_mask, za)
            q_ids = self._generate_question(c_ids, c_mask, gen_c_a_mask, zq)

            return q_ids, gen_c_a_start_positions, gen_c_a_end_positions

    def validation_step(self, batch, batch_idx):
        def generate_qa_from_posterior(question_ids, context_ids, c_a_mask):
            with torch.no_grad():
                c_mask = return_attention_mask(context_ids, self.pad_token_id)
                q_mask = return_attention_mask(question_ids, self.pad_token_id)

                zq, _, _, za, _, _ = self._encode_latent_posteriors(
                    question_ids, q_mask, context_ids, c_mask, c_a_mask)

                """ Generation """
                gen_c_a_mask, gen_c_a_start_positions, gen_c_a_end_positions, start_logits, end_logits = \
                    self._generate_answer(context_ids, c_mask, za)
                question_ids = self._generate_question(context_ids, c_mask, gen_c_a_mask, zq)

                return question_ids, gen_c_a_start_positions, gen_c_a_end_positions, start_logits, end_logits

        def to_string(index, tokenizer_):
            tok_tokens = tokenizer_.convert_ids_to_tokens(index)
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(tokenizer_.pad_token, "")
            tok_text = tok_text.replace(tokenizer_.sep_token, "")
            # tok_text = tok_text.replace(tokenizer.cls_token, "")
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            return tok_text

        RawResult = collections.namedtuple("RawResult",
                                           ["unique_id", "start_logits", "end_logits"])

        tokenizer = self.tokenizer

        posterior_qa_results = []
        qg_results = {}
        real_question_dict = {}

        assert isinstance(self.trainer.val_dataloaders[0].dataset, CustomDataset), \
            "ERROR: validation set is not constructed from `CustomDataset` class"

        # only one validation set
        all_preprocessed_examples = self.trainer.val_dataloaders[0].dataset.all_preprocessed_examples

        _, q_ids, c_ids, a_mask, _, no_q_start_positions, no_q_end_positions = batch
        batch_size = c_ids.size(0)
        batch_q_ids = q_ids.cpu().tolist()

        batch_posterior_q_ids, batch_posterior_start, batch_posterior_end, batch_start_logits, batch_end_logits, \
            = generate_qa_from_posterior(q_ids, c_ids, a_mask)

        # Convert posterior tensors to Python list
        batch_posterior_q_ids, batch_posterior_start, batch_posterior_end = \
            batch_posterior_q_ids.cpu().tolist(), batch_posterior_start.cpu().tolist(), \
            batch_posterior_end.cpu().tolist()

        for i in range(batch_size):
            self.example_idx += 1
            posterior_start_logits = batch_start_logits[i].detach().cpu().tolist()
            posterior_end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = all_preprocessed_examples[self.example_idx]
            unique_id = int(eval_feature.unique_id)

            real_question = to_string(batch_q_ids[i], tokenizer)
            posterior_question = to_string(batch_posterior_q_ids[i], tokenizer)

            qg_results[unique_id] = posterior_question
            real_question_dict[unique_id] = real_question
            posterior_qa_results.append(RawResult(unique_id=unique_id,
                                                  start_logits=posterior_start_logits,
                                                  end_logits=posterior_end_logits))

        return {"posterior_qa": posterior_qa_results, "real_questions": real_question_dict, "qg_results": qg_results}

    def validation_epoch_end(self, outputs: Dict):
        # only one validation set
        all_text_examples = self.trainer.val_dataloaders[0].dataset.all_text_examples
        all_preprocessed_examples = self.trainer.val_dataloaders[0].dataset.all_preprocessed_examples

        posterior_qa_results = list(itertools.chain.from_iterable([out["posterior_qa"] for out in outputs]))
        real_question_dict = {k: v for out in outputs for k, v in out["real_questions"].items()}
        qg_results = {k: v for out in outputs for k, v in out["qg_results"].items()}

        posterior_predictions = extract_predictions_to_dict(all_text_examples, all_preprocessed_examples,
                                                            posterior_qa_results,
                                                            n_best_size=20, max_answer_length=30, do_lower_case=True,
                                                            verbose_logging=False, version_2_with_negative=False,
                                                            null_score_diff_threshold=0, noq_position=True)
        posterior_ret = evaluate(self.dev_dataset, posterior_predictions)
        bleu = eval_qg(real_question_dict, qg_results)

        metrics = {"f1": posterior_ret["f1"], "exact_match": posterior_ret["exact_match"], "bleu": bleu}
        self.log_dict(metrics, prog_bar=True)

        # Log to file
        log_str = "f1: {:.4f} - em: {:.4f} - bleu: {:.4f}".format(posterior_ret["f1"], posterior_ret["exact_match"], bleu)
        with open(self.eval_metrics_log_file, "a") as f:
            f.write(log_str + "\n\n")

        self.example_idx = -1 # reset example index for next validation loop

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        if self.optimizer_algorithm == "sgd":
            optimizer = optim.SGD(params, lr=self.lr, momentum=0.9, nesterov=False)
        elif self.optimizer_algorithm == "adam":
            optimizer = optim.Adam(params, lr=self.lr)
        elif self.optimizer_algorithm == "adamw":
            optimizer = optim.AdamW(params, lr=self.lr)
        else:
            optimizer = additional_optim.SWATS(params, lr=self.lr, nesterov=False)
        return optimizer
