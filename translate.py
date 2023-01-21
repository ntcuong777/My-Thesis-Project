import argparse
import pickle
import h5py
import random

import torch
import numpy as np
import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
from infohcvae.model.bert_qag_cvae import BertQAGConditionalVae
from infohcvae.model.model_utils import return_attention_mask
from infohcvae.squad_utils import (
    read_squad_examples, convert_examples_to_features_answer_id_for_generation,
)


def return_seq_lengths(mask):
    return torch.sum(mask, dim=1)


def post_process(q_ids, start_positions, end_positions, c_ids, pad_token_id, total_max_len=448):
    """
       concatenate question and context for BERT QA model:
       [CLS] Question [SEP] Context [SEP]
    """
    batch_size = q_ids.size(0)
    # exclude CLS token in c_ids
    c_ids = c_ids[:, 1:]
    start_positions = start_positions - 1
    end_positions = end_positions - 1

    q_lengths = return_seq_lengths(return_attention_mask(q_ids, pad_token_id))
    c_lengths = return_seq_lengths(return_attention_mask(c_ids, pad_token_id))

    all_input_ids = []
    all_seg_ids = []
    for i in range(batch_size):
        q_length = int(q_lengths[i].item())
        c_length = int(c_lengths[i].item())
        q = q_ids[i, :q_length]  # exclude pad tokens
        c = c_ids[i, :c_length]  # exclude pad tokens

        # input ids
        pads = torch.zeros((total_max_len - q_length - c_length), device=q_ids.device, dtype=torch.long)
        input_ids = torch.cat([q, c, pads], dim=0)
        all_input_ids.append(input_ids)

        # segment ids
        zeros = torch.zeros_like(q)
        ones = torch.ones_like(c)
        seg_ids = torch.cat([zeros, ones, pads], dim=0)
        all_seg_ids.append(seg_ids)

        start_positions[i] = start_positions[i] + q_length
        end_positions[i] = end_positions[i] + q_length

    all_input_ids = torch.stack(all_input_ids, dim=0)
    all_seg_ids = torch.stack(all_seg_ids, dim=0)
    all_input_mask = (all_input_ids != 0).byte()

    return all_input_ids, all_seg_ids, all_input_mask, start_positions, end_positions


def main(gen_args):
    tokenizer = AutoTokenizer.from_pretrained(gen_args.base_model)
    gen_args.tokenizer = tokenizer
    pad_token_id = tokenizer.pad_token_id

    device = torch.cuda.current_device()
    vae = BertQAGConditionalVae.load_from_checkpoint(gen_args.checkpoint)
    vae.eval()
    vae = vae.to(device)

    data_loader = None
    if not gen_args.load_saved_dataloader:
        # Add shuffling functionality if wanting to use a small percentage of data correctly
        features = []
        if gen_args.squad:
            examples = read_squad_examples(gen_args.data_file, is_training=True, debug=gen_args.debug)
            features = convert_examples_to_features_answer_id_for_generation(
                examples, tokenizer=tokenizer, max_context_length=gen_args.max_c_len, doc_stride=128)
            full_data = {
                "example": examples,
                "features": features
            }
            with open('full_processed_data.pickle', 'wb') as handle:
                pickle.dump(full_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # else:
        #     examples = read_examples(gen_args.data_file, is_training=True, debug=gen_args.debug)
        #     features = convert_examples_to_harv_features(examples, tokenizer=tokenizer,
        #                                                  max_seq_length=gen_args.max_c_len,
        #                                                  max_query_length=0, doc_stride=128,
        #                                                  is_training=True)

        perm = np.arange(0, len(features))
        np.random.shuffle(perm)
        perm = perm.tolist()[:int(len(features) * gen_args.ratio)]
        features_tmp = [features[i] for i in perm]
        all_c_ids = torch.tensor([f.c_ids for f in features_tmp], dtype=torch.long)
        data = TensorDataset(all_c_ids)
        data_loader = DataLoader(data, shuffle=False, batch_size=gen_args.batch_size)
        print("Dataset length = " + str(len(data_loader.dataset)))
        torch.save(data_loader, os.path.join(gen_args.dataloader_dir, "gen_loader.pt"))
    else:
        data_loader = torch.load(os.path.join(gen_args.dataloader_dir, "gen_loader.pt"))
        print("Dataset length = " + str(len(data_loader.dataset)))

    with h5py.File(gen_args.output_file, "a") as fdata:
        input_ids_set = fdata.create_dataset(
            "qas/input_ids", (len(data_loader.dataset) * gen_args.k, gen_args.total_max_len),
            chunks=(100, gen_args.total_max_len))
        input_masks_set = fdata.create_dataset(
            "qas/input_masks", (len(data_loader.dataset) * gen_args.k, gen_args.total_max_len),
            chunks=(100, gen_args.total_max_len))
        segment_ids_set = fdata.create_dataset(
            "qas/segment_ids", (len(data_loader.dataset) * gen_args.k, gen_args.total_max_len),
            chunks=(100, gen_args.total_max_len))
        start_positions_set = fdata.create_dataset(
            "qas/start_positions", (len(data_loader.dataset) * gen_args.k,), chunks=(1000,))
        end_positions_set = fdata.create_dataset(
            "qas/end_positions", (len(data_loader.dataset) * gen_args.k,), chunks=(1000,))

        # new_features = []
        qa_text = None
        if gen_args.out_qa_json is not None:
            qa_text = dict({"data": []})

        num_steps_to_run = len(data_loader.dataset)
        print("Num steps to run: {:d}".format(num_steps_to_run))
        qa_idx = 0
        for batch in tqdm(data_loader, total=len(data_loader)):
            if num_steps_to_run == 0:
                break

            num_steps_to_run = num_steps_to_run - 1

            c_ids = batch[0]
            c_len = return_seq_lengths(return_attention_mask(c_ids, pad_token_id))
            max_c_len = int(torch.max(c_len).item())
            c_ids = c_ids[:, :max_c_len].to(device)

            c_texts = [gen_args.tokenizer.decode(c_ids[idx]) for idx in range(c_ids.size(0))]

            # sample latent variable K times
            with torch.no_grad():
                # c_ids = (N, seq_len)
                repeated_c_ids = c_ids.unsqueeze(1).repeat(1, gen_args.k, 1).view(gen_args.batch_size * gen_args.k, -1)
                batch_q_ids, batch_start, batch_end = vae.generate_qa_from_prior(repeated_c_ids)
                # batch_q_ids = batch_q_ids.view(gen_args.batch_size, gen_args.k, -1) # (N, k, seq_len)
                # batch_start = batch_start.view(gen_args.batch_size, gen_args.k, -1) # (N, k)
                # batch_end = batch_end.view(gen_args.batch_size, gen_args.k, -1) # (N, k)

                if gen_args.output_text and gen_args.out_qa_json is not None:  # out QA text to json
                    for idx in range(batch_q_ids.size(0)):
                        q_ids, start_pos, end_pos = batch_q_ids[idx], batch_start[idx], batch_end[idx]
                        q_text = gen_args.tokenizer.decode(q_ids)
                        ans_text = gen_args.tokenizer.decode(repeated_c_ids[idx, start_pos:end_pos])
                        qa_text["data"].append({"context": c_texts[idx // gen_args.k],
                                                "question": q_text, "answer": ans_text})

                all_input_ids, all_seg_ids, \
                    all_input_mask, all_start, all_end = post_process(
                        batch_q_ids, batch_start, batch_end, repeated_c_ids,
                        pad_token_id, total_max_len=gen_args.total_max_len)

                for i in range(repeated_c_ids.size(0)):
                    input_ids_set[qa_idx, :] = all_input_ids[i].cpu()
                    input_masks_set[qa_idx, :] = all_input_mask[i].cpu()
                    segment_ids_set[qa_idx, :] = all_seg_ids[i].cpu()
                    start_positions_set[qa_idx] = all_start[i].cpu()
                    end_positions_set[qa_idx] = all_end[i].cpu()
                    qa_idx += 1
            # for _ in range(gen_args.k):
            #     with torch.no_grad():
            #         batch_q_ids, batch_start, batch_end = vae.generate_qa_from_prior(c_ids)
            #
            #         if gen_args.output_text and gen_args.out_qa_json is not None:  # out QA text to json
            #             for idx in range(batch_q_ids.size(0)):
            #                 q_ids, start_pos, end_pos = batch_q_ids[idx], batch_start[idx], batch_end[idx]
            #                 q_text = gen_args.tokenizer.decode(q_ids)
            #                 ans_text = gen_args.tokenizer.decode(c_ids[idx, start_pos:end_pos])
            #                 qa_text["data"].append({"context": c_texts[idx], "question": q_text, "answer": ans_text})
            #
            #         all_input_ids, all_seg_ids, \
            #             all_input_mask, all_start, all_end = post_process(
            #                 batch_q_ids, batch_start, batch_end, c_ids, pad_token_id, total_max_len=gen_args.total_max_len)
            #
            #     for i in range(c_ids.size(0)):
            #         input_ids_set[qa_idx, :] = all_input_ids[i].cpu()
            #         input_masks_set[qa_idx, :] = all_input_mask[i].cpu()
            #         segment_ids_set[qa_idx, :] = all_seg_ids[i].cpu()
            #         start_positions_set[qa_idx] = all_start[i].cpu()
            #         end_positions_set[qa_idx] = all_end[i].cpu()
            #         qa_idx += 1

    # For outputting text
    if gen_args.output_text:
        import json
        dir_name = os.path.dirname(gen_args.out_qa_json)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(gen_args.out_qa_json, "wt") as f:
            json.dump(qa_text, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--squad', dest='squad', action='store_true', help="whether to generate QA from SQuAD context")
    parser.add_argument("--load_saved_dataloader", dest="load_saved_dataloader", action="store_true", default=False)
    parser.add_argument("--output_text", dest="output_text", action="store_true", default=False)

    parser.add_argument("--seed", default=3163, type=int)
    parser.add_argument("--base_model", default='bert-base-uncased', type=str)
    parser.add_argument("--max_c_len", default=384, type=int, help="max context length")
    parser.add_argument("--max_q_len", default=64, type=int, help="max query length")

    parser.add_argument("--batch_size", default=64, type=int, help="batch_size")
    parser.add_argument("--data_file", default="./data/squad/train-v1.1.json", type=str)
    parser.add_argument("--checkpoint", default="./save/vae-checkpoint/best_f1_model.pt", type=str,
                        help="checkpoint for vae model")
    parser.add_argument("--output_file", default="./data/1.0_squad_10x_features.h5", type=str)
    parser.add_argument("--out_qa_json", default="./data/generated_qas.json", type=str)
    parser.add_argument("--dataloader_dir", default="./save/dataloader", type=str)

    parser.add_argument("--ratio", default=1.0, type=float)
    parser.add_argument("--k", default=10, type=int, help="the number of QA pairs for each paragraph")

    args = parser.parse_args()

    args.total_max_len = args.max_c_len + args.max_q_len

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    pl.seed_everything(args.seed)

    # set dataloader dir
    if not args.load_saved_dataloader:
        dataloader_dir = args.dataloader_dir
        os.makedirs(dataloader_dir, exist_ok=True)
        args.dataloader_dir = os.path.abspath(dataloader_dir)

    main(args)
