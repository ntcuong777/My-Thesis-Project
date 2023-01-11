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
from evaluation.qgevalcap.eval import eval_qg
from infohcvae.squad_utils import evaluate, extract_predictions_to_dict


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

    def forward(self, c_ids, q_ids, a_mask):
        za_logits, za, zq_mu, zq_logvar, zq \
            = self.posterior_encoder(c_ids, q_ids, a_mask, return_distribution_parameters=True)

        # answer decoding
        start_logits, end_logits = self.answer_decoder(c_ids, za)
        # question decoding
        q_logits, (q_mean_emb, a_mean_emb) = self.question_decoder(c_ids, q_ids, a_mask, zq, return_qa_mean_embeds=True)

        return dict({"latent_codes": (za_logits, za, zq_mu, zq_logvar, zq),
                     "answer_rec": (start_logits, end_logits),
                     "question_rec": (q_logits, q_mean_emb, a_mean_emb)})

    def training_step(self, batch, batch_idx):
        c_ids, q_ids, a_mask, start_positions, end_positions = batch
        outputs = self.forward(c_ids, q_ids, a_mask)
        zq_mu, zq_logvar, zq, za_logits, za = outputs["latent_codes"]
        start_logits, end_logits = outputs["answer_rec"]
        q_logits, q_mean_emb, a_mean_emb = outputs["question_rec"]

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
        self.log_dict(current_losses, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        def to_string(index, tokenizer):
            tok_tokens = tokenizer.convert_ids_to_tokens(index)
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace("[PAD]", "")
            tok_text = tok_text.replace("[SEP]", "")
            tok_text = tok_text.replace("[CLS]", "")
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            return tok_text

        # TODO: Complete the validation step

        c_ids, q_ids, a_mask, _, _, examples, features = batch

        RawResult = collections.namedtuple("RawResult", ["start_logits", "end_logits"])

        posterior_qa_results = []
        qg_results = {}
        res_dict = {}

        #### Start evaluation
        batch_size = c_ids.size(0)
        batch_q_ids = q_ids.cpu().tolist()

        batch_posterior_q_ids, \
            batch_posterior_start, batch_posterior_end, \
            posterior_zq = self.generate_qa_from_posterior(c_ids, q_ids, a_mask)

        batch_start_logits, batch_end_logits \
            = self.generate_answer_logits_from_posterior(c_ids, q_ids, a_mask)

        # Convert posterior tensors to Python list
        batch_posterior_q_ids, \
            batch_posterior_start, batch_posterior_end = \
            batch_posterior_q_ids.cpu().tolist(), \
                batch_posterior_start.cpu().tolist(), batch_posterior_end.cpu().tolist()

        for i in range(batch_size):
            posterior_start_logits = batch_start_logits[i].detach().cpu().tolist()
            posterior_end_logits = batch_end_logits[i].detach().cpu().tolist()
            unique_id = int(features[i].unique_id)

            real_question = to_string(batch_q_ids[i], self.tokenizer)
            posterior_question = to_string(batch_posterior_q_ids[i], self.tokenizer)

            qg_results[unique_id] = posterior_question
            res_dict[unique_id] = real_question
            posterior_qa_results.append(RawResult(start_logits=posterior_start_logits,
                                                  end_logits=posterior_end_logits))

        # Evaluate metrics
        import json
        with open(self.args.dev_dir) as f:
            dataset_json = json.load(f)
            dataset = dataset_json["data"]

        posterior_predictions = extract_predictions_to_dict(examples, features, self.posterior_qa_results,
                                                            n_best_size=20, max_answer_length=30, do_lower_case=True,
                                                            verbose_logging=False, version_2_with_negative=False,
                                                            null_score_diff_threshold=0, noq_position=True)
        posterior_ret = evaluate(dataset, posterior_predictions)
        bleu = eval_qg(res_dict, qg_results)

        metrics = {"f1": posterior_ret["f1"], "exact_match": posterior_ret["exact_match"], "bleu": bleu}
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

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
