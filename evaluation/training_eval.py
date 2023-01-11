import collections
import json
import os
from transformers import MobileBertTokenizer
from tqdm import tqdm

from qgevalcap.eval import eval_qg
from infohcvae.squad_utils import evaluate, extract_predictions_to_dict
from infohcvae.utils import batch_to_device

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

class Result(object):
    def __init__(self,
                 context,
                 real_question,
                 posterior_question,
                 prior_question,
                 real_answer,
                 posterior_answer,
                 prior_answer,
                 posterior_z_prob,
                 prior_z_prob):
        self.context = context
        self.real_question = real_question
        self.posterior_question = posterior_question
        self.prior_question = prior_question
        self.real_answer = real_answer
        self.posterior_answer = posterior_answer
        self.prior_answer = prior_answer
        self.posterior_z_prob = posterior_z_prob
        self.prior_z_prob = prior_z_prob


def eval_vae(args, trainer, eval_data):
    trainer.set_eval_mode(True)

    tokenizer = MobileBertTokenizer.from_pretrained(args.huggingface_model)
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])

    eval_loader, eval_examples, eval_features = eval_data

    posterior_qa_results = []
    # prior_qa_result = []
    qg_results = {}
    res_dict = {}
    example_index = -1

    for batch in tqdm(eval_loader, desc="Eval iter", leave=False, position=4):
        c_ids, q_ids, a_ids, _, _ = batch_to_device(batch, args.device)
        batch_size = c_ids.size(0)
        batch_q_ids = q_ids.cpu().tolist()

        batch_posterior_q_ids, \
        batch_posterior_start, batch_posterior_end, \
        posterior_zq = trainer.generate_qa_from_posterior(c_ids, q_ids, a_ids)

        batch_start_logits, batch_end_logits \
        = trainer.generate_answer_logits_from_posterior(c_ids, q_ids, a_ids)

        # in this case: prior_zq == posterior_zq
        # batch_prior_start_logits, batch_prior_end_logits = \
        #     trainer.generate_prior_answer_logits(c_ids, q_ids, a_ids)

        # Convert posterior tensors to Python list
        batch_posterior_q_ids, \
        batch_posterior_start, batch_posterior_end = \
        batch_posterior_q_ids.cpu().tolist(), \
        batch_posterior_start.cpu().tolist(), batch_posterior_end.cpu().tolist()

        for i in range(batch_size):
            example_index += 1
            posterior_start_logits = batch_start_logits[i].detach().cpu().tolist()
            posterior_end_logits = batch_end_logits[i].detach().cpu().tolist()
            # prior_start_logits = batch_prior_start_logits[i].detach().cpu().tolist()
            # prior_end_logits = batch_prior_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index]
            unique_id = int(eval_feature.unique_id)

            real_question = to_string(batch_q_ids[i], tokenizer)
            posterior_question = to_string(batch_posterior_q_ids[i], tokenizer)

            qg_results[unique_id] = posterior_question
            res_dict[unique_id] = real_question
            posterior_qa_results.append(RawResult(unique_id=unique_id,
                                        start_logits=posterior_start_logits,
                                        end_logits=posterior_end_logits))
            # prior_qa_result.append(RawResult(unique_id=unique_id,
            #                         start_logits=prior_start_logits,
            #                         end_logits=prior_end_logits))

    posterior_predictions = extract_predictions_to_dict(eval_examples, eval_features, posterior_qa_results, n_best_size=20,
                                max_answer_length=30, do_lower_case=True,
                                verbose_logging=False,
                                version_2_with_negative=False,
                                null_score_diff_threshold=0,
                                noq_position=True)

    posterior_out_pred_file = os.path.join(args.model_dir, "posterior_pred.json")
    with open(posterior_out_pred_file, "w") as f:
        json.dump(posterior_predictions, f, indent=4, ensure_ascii=True)

    with open(args.dev_dir) as f:
        dataset_json = json.load(f)
        dataset = dataset_json["data"]
    with open(os.path.join(args.model_dir, "posterior_pred.json")) as prediction_file:
        posterior_predictions = json.load(prediction_file)
    # with open(os.path.join(args.model_dir, "prior_pred.json")) as prediction_file:
    #     prior_predictions = json.load(prediction_file)
    posterior_ret = evaluate(dataset, posterior_predictions)
    # prior_ret = evaluate(dataset, posterior_predictions)
    bleu = eval_qg(res_dict, qg_results)

    # return posterior_ret, prior_ret, bleu
    return posterior_ret, bleu